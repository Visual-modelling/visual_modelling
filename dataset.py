__author__ = "Jumperkables"

import os, sys, random, math, argparse
import getpass

from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import cv2
import pandas as pd
from torchvision.utils import save_image

import tools.utils as utils


###########################################################################################
# Utility methods
###########################################################################################
# Image reading methods
def read_ims_binary(frames):
    raise Exception("Not implemented binary image read yet")
    return frames
    #old pillow version return([ (ToTensor()(Image.open(frame.replace('jumperkables', getpass.getuser()) ))>0).float() for frame in frames ])

def read_ims_greyscale(frames):
    home_dir = os.path.expanduser("~").split("/")[1]
    #import ipdb; ipdb.set_trace()
    frames = [ torch.from_numpy(cv2.imread(frame.replace('jumperkables', getpass.getuser()).replace("/home/", f"/{home_dir}/" ), cv2.IMREAD_GRAYSCALE)) for frame in frames]
    #old pillow version frames = [ (ToTensor()(Image.open(frame.replace('jumperkables', getpass.getuser()) ))>0).float() for frame in frames ]
    return frames




###########################################################################################
# Dataset Classes
###########################################################################################
class Dataset_from_raw(Dataset):
    """
    Dataset object that processes from raw frames:
    Designed to work with dataset of depth 3.
    root_dir -> clip_0,...clip_999 -> frame_000.png, frame_001.png
    """
    def __init__(self, dataset_path, args):
        self.mode = "train"
        self.args = args
        data = self.read_frames(dataset_path)
        self.train_dict, self.valid_dict = self.train_val_split(data, args.split_condition)
        self.train_dict, self.valid_dict, self.self_out_dict = self.prepare_dicts(self.train_dict, self.valid_dict)
        self.set_mode("train")
        # Sort Image reading method
        img_read_method_switch = {
            'binary'    : read_ims_binary,
            'greyscale' : read_ims_greyscale,
            'RGB'       : None
        }
        if args.img_type == "RGB":
            raise Exception("RGB image reading not implemented")
        self.img_read_method = img_read_method_switch[args.img_type]

    def set_mode(self, mode):
        """
        Jump between training/validation mode
        """
        self.mode = mode
        if self.mode == 'train':
            self.current_data_dict = self.train_dict
        elif self.mode == 'valid':
            self.current_data_dict = self.valid_dict
        elif self.mode == "self_out":
            self.current_data_dict = self.self_out_dict
        else:
            raise ValueError(f"Mode {mode} invalid")

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
        if self.mode is not "self_out":
            #import ipdb; ipdb.set_trace()
            data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
            positions = torch.stack([ pos['positions'] for pos in data.values() ])
            positions, gt_positions = positions[:self.args.in_no], positions[self.args.in_no:]
            frames = [ frm['frame_paths'] for frm in data.values() ]
            if self.args.segmentation:
                # If its a semgentation task, read the final ground_truth frame in its segmentation mask form instead
                #import ipdb; ipdb.set_trace()
                frames[-1] = frames[-2].replace("hudson_true_3d_default", "hudson_true_3d_default_mask")
            frames = self.img_read_method(frames)
            frames = torch.stack(frames, dim=0)
            frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
            return (frames, gt_frames)
        else:
            data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
            frames = [ frm['frame_paths'] for frm in data.values() ]
            frames = self.img_read_method(frames)
            frames = torch.stack(frames, dim=0)
            start_frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
            vid_name = [ frm['vid'] for frm in data.values() ][0] 
            return (start_frames, gt_frames, vid_name)           

    def prepare_dicts(self, train_dict, valid_dict):
        # Training
        iter_len = self.args.in_no + self.args.out_no
        new_train = {}
        counter = 0
        for idx, clip in enumerate(train_dict.values()):
            clip_len = len(clip.keys())
            for x in range(0, clip_len-(clip_len % iter_len), iter_len):
                new_train[counter] = { y:clip[y] for y in range(x,x+iter_len) }
                counter+=1
        #self.train_dict = new_train

        # Validation
        new_valid = {}
        new_self_output = {}
        counter = 0
        for idx, clip in enumerate(valid_dict.values()):
            clip_len = len(clip.keys())
            new_self_output[idx] = clip
            for x in range(0, clip_len-(clip_len % iter_len), iter_len):
                new_valid[counter] = { y:clip[y] for y in range(x,x+iter_len) }
                counter+=1
        return new_train, new_valid, new_self_output

    def train_val_split(self, data, condition):
        """
        Return train and validation data dictionaries
        """
        #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
        if self.args.dset_sze != -1:
            data = dict( list( data.items()[self.args.dset_sze:] ) )
        # See the argparse in main for a description of splitting functions
        if condition[:8] == "tv_ratio":
            tv_ratio = condition[9:].split('-')
            tv_ratio = float(tv_ratio[0])/( float(tv_ratio[0]) + float(tv_ratio[1]) )
            train_dict, valid_dict = utils.split_dict_ratio(data, tv_ratio)
            return train_dict, valid_dict
        else:
            raise ValueError(f"Condition: {condition} not recognised")

    def read_frames(self, dataset_path):
        """
        Read in the raw images as frames into a dictionary
        Returns:
        {
            'clip_name':{0: {frame_0_data}}
            ...
        }
        """
        frame_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), dataset_path)
        #vids = os.listdir(frame_path) # Remove

        vids = []
        path = os.path.normpath(frame_path)
        for root,dirs,files in os.walk(frame_path, topdown=True):
            for dyr in dirs:
                depth_test = os.listdir( os.path.join( root, dyr ) )
                if any( (fyle.startswith("frame") and (fyle.endswith(".png") or fyle.endswith(".jpg")) ) for fyle in depth_test):
                    vids.append( os.path.join(root, dyr) )

        ###########################
        ## SHOULD be outdated
        ## Remove all excess files
        #try:
        #    vids.remove('config.yml')
        #    vids.remove('config2.yml')
        #except:
        #    pass
        #vids = [vid for vid in vids if not (vid.endswith('.pickle') or vid.endswith('.jsonl'))]
        #vids.sort()
        ###########################
        
        total_data = { vid:os.path.join(frame_path, vid) for vid in vids }
        return_dataset = {}
        for vidx, vid_path in total_data.items():
            print(vidx)
            vid_name = vid_path.split('/')[-1]
            try:
                positions = utils.read_csv(os.path.join(vid_path, 'positions.csv'))
            except FileNotFoundError:
                print(f"{vid_path} not found.\nCreating dummy 'positions' information")
                

                frame_cnt = len(os.listdir(vid_path))
                positions = {
                    'timestep': [i for i in range(frame_cnt)],
                    'x': [0]*frame_cnt,
                    'y': [0]*frame_cnt
                }
                positions = pd.DataFrame(data=positions)
            indexs = torch.tensor(positions.values)[:,:1].long()
            positions = torch.tensor(positions.values)[:,1:]    # Remove the useless frame index for now
            frames = os.listdir(vid_path)
            frames.sort()
            exclude = ['config.yml','config2.yml','positions.csv','simulation.gif','mask']
            frames = [ os.path.join(vid_path, frame) for frame in frames if frame not in exclude]
            frame_paths = frames

            return_dataset[vidx] = {x:{
                            "vid":vid_name,
                            "vid_path":vid_path,
                            "frame_idx":indexs[x],
                            "frame_paths":frame_paths[x],
                            "positions":positions[x],
                            "frame":frames[x]} 
                        for x in range(len(frames))}
        return return_dataset
 






class HDMB_classification(Dataset):
    """
    Designed to work with dataset of depth 3.
    root_dir -> clip_0,...clip_999 -> frame_000.png, frame_001.png
    """
    def __init__(self, dataset_path, args):
        self.mode = "train"
        self.args = args
        data = self.read_frames(dataset_path)
        self.train_dict, self.valid_dict = self.train_val_split(data, args.split_condition)
        self.train_dict, self.valid_dict, self.self_out_dict = self.prepare_dicts(self.train_dict, self.valid_dict)
        self.set_mode("train")
        # Sort Image reading method
        img_read_method_switch = {
            'binary'    : read_ims_binary,
            'greyscale' : read_ims_greyscale,
            'RGB'       : None
        }
        if args.img_type == "RGB":
            raise Exception("RGB image reading not implemented")
        self.img_read_method = img_read_method_switch[args.img_type]

    def set_mode(self, mode):
        """
        Jump between training/validation mode
        """
        self.mode = mode
        if self.mode == 'train':
            self.current_data_dict = self.train_dict
        elif self.mode == 'valid':
            self.current_data_dict = self.valid_dict
        elif self.mode == "self_out":
            self.current_data_dict = self.self_out_dict
        else:
            raise ValueError(f"Mode {mode} invalid")

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
        if self.mode is not "self_out":
            #import ipdb; ipdb.set_trace()
            data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
            positions = torch.stack([ pos['positions'] for pos in data.values() ])
            positions, gt_positions = positions[:self.args.in_no], positions[self.args.in_no:]
            frames = [ frm['frame_paths'] for frm in data.values() ]
            if self.args.segmentation:
                # If its a semgentation task, read the final ground_truth frame in its segmentation mask form instead
                #import ipdb; ipdb.set_trace()
                frames[-1] = frames[-2].replace("hudson_true_3d_default", "hudson_true_3d_default_mask")
            frames = self.img_read_method(frames)
            frames = torch.stack(frames, dim=0)
            frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
            return (frames, gt_frames)
        else:
            data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
            frames = [ frm['frame_paths'] for frm in data.values() ]
            frames = self.img_read_method(frames)
            frames = torch.stack(frames, dim=0)
            start_frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
            vid_name = [ frm['vid'] for frm in data.values() ][0] 
            return (start_frames, gt_frames, vid_name)           

    def prepare_dicts(self, train_dict, valid_dict):
        # Training
        iter_len = self.args.in_no + self.args.out_no
        new_train = {}
        counter = 0
        for idx, clip in enumerate(train_dict.values()):
            clip_len = len(clip.keys())
            for x in range(0, clip_len-(clip_len % iter_len), iter_len):
                new_train[counter] = { y:clip[y] for y in range(x,x+iter_len) }
                counter+=1
        #self.train_dict = new_train

        # Validation
        new_valid = {}
        new_self_output = {}
        counter = 0
        for idx, clip in enumerate(valid_dict.values()):
            clip_len = len(clip.keys())
            new_self_output[idx] = clip
            for x in range(0, clip_len-(clip_len % iter_len), iter_len):
                new_valid[counter] = { y:clip[y] for y in range(x,x+iter_len) }
                counter+=1
        return new_train, new_valid, new_self_output

    def train_val_split(self, data, condition):
        """
        Return train and validation data dictionaries
        """
        #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
        if self.args.dset_sze != -1:
            data = dict( list( data.items()[self.args.dset_sze:] ) )
        # See the argparse in main for a description of splitting functions
        if condition[:8] == "tv_ratio":
            tv_ratio = condition[9:].split('-')
            tv_ratio = float(tv_ratio[0])/( float(tv_ratio[0]) + float(tv_ratio[1]) )
            train_dict, valid_dict = utils.split_dict_ratio(data, tv_ratio)
            return train_dict, valid_dict
        else:
            raise ValueError(f"Condition: {condition} not recognised")

    def read_frames(self, dataset_path):
        """
        Read in the raw images as frames into a dictionary
        Returns:
        {
            'clip_name':{0: {frame_0_data}}
            ...
        }
        """
        frame_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), dataset_path)
        #vids = os.listdir(frame_path) # Remove

        vids = []
        path = os.path.normpath(frame_path)
        for root,dirs,files in os.walk(frame_path, topdown=True):
            for dyr in dirs:
                depth_test = os.listdir( os.path.join( root, dyr ) )
                if any( (fyle.startswith("frame") and (fyle.endswith(".png") or fyle.endswith(".jpg")) ) for fyle in depth_test):
                    vids.append( os.path.join(root, dyr) )

        ###########################
        ## SHOULD be outdated
        ## Remove all excess files
        #try:
        #    vids.remove('config.yml')
        #    vids.remove('config2.yml')
        #except:
        #    pass
        #vids = [vid for vid in vids if not (vid.endswith('.pickle') or vid.endswith('.jsonl'))]
        #vids.sort()
        ###########################
        
        total_data = { vid:os.path.join(frame_path, vid) for vid in vids }
        return_dataset = {}
        for vidx, vid_path in total_data.items():
            print(vidx)
            vid_name = vid_path.split('/')[-1]
            try:
                positions = utils.read_csv(os.path.join(vid_path, 'positions.csv'))
            except FileNotFoundError:
                print(f"{vid_path} not found.\nCreating dummy 'positions' information")
                

                frame_cnt = len(os.listdir(vid_path))
                positions = {
                    'timestep': [i for i in range(frame_cnt)],
                    'x': [0]*frame_cnt,
                    'y': [0]*frame_cnt
                }
                positions = pd.DataFrame(data=positions)
            indexs = torch.tensor(positions.values)[:,:1].long()
            positions = torch.tensor(positions.values)[:,1:]    # Remove the useless frame index for now
            frames = os.listdir(vid_path)
            frames.sort()
            exclude = ['config.yml','config2.yml','positions.csv','simulation.gif','mask']
            frames = [ os.path.join(vid_path, frame) for frame in frames if frame not in exclude]
            frame_paths = frames

            return_dataset[vidx] = {x:{
                            "vid":vid_name,
                            "vid_path":vid_path,
                            "frame_idx":indexs[x],
                            "frame_paths":frame_paths[x],
                            "positions":positions[x],
                            "frame":frames[x]} 
                        for x in range(len(frames))}
        return return_dataset
 







###########################################################################################
# Code adapted from: https://gist.github.com/tencia/afb129122a64bde3bd0c

# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################
class MMNIST(Dataset):
    def __init__(self, dataset_path, args):
        self.args = args
        self.generate_moving_mnist((64,64), 100, [1000,2000,3000], 28, [1,2,3])
        sys.exit()
        self.mode = "train"
        total_dset = np.load( os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), dataset_path))
        total_dset = total_dset['arr_0']

        self.train_dict, self.valid_dict = self.train_val_split(data, args.split_condition)
        self.train_dict, self.valid_dict, self.self_out_dict= self.prepare_dicts()
        self.set_mode("train")

        self.current_data_dict = self.train_dict

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
        frames = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
        frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
        return (torch.from_numpy(frames.squeeze(1)), torch.from_numpy(gt_frames.squeeze(1)))

    def set_mode(self, mode):
        """
        Jump between training/validation mode
        """
        self.mode = mode
        if self.mode == 'train':
            self.current_data_dict = self.train_dict
        elif self.mode == 'valid':
            self.current_data_dict = self.valid_dict
        elif self.mode == "self_out":
            self.current_data_dict = self.self_out_dict 

    def train_val_split(self, data, condition):
        """
        Return train and validation data dictionaries
        """
        if self.args.dset_sze != -1:
            data = dict( list( data.items()[self.args.dset_sze:] ) )
        # See the argparse in main for a description of splitting functions
        if condition[:8] == "tv_ratio":
            tv_ratio = condition[9:].split('-')
            tv_ratio = float(tv_ratio[0])/( float(tv_ratio[0]) + float(tv_ratio[1]) )
            train_dict, valid_dict = utils.split_dict_ratio(data, tv_ratio)
            return train_dict, valid_dict
        else:
            raise ValueError(f"Condition: {condition} not recognised")

    def prepare_dicts(self, train_dict, valid_dict):
        # Training
        #import ipdb; ipdb.set_trace()
        iter_len = self.args.in_no + self.args.out_no
        new_train = {}
        counter = 0
        for idx, clip in enumerate(train_dict.values()):
            clip_len = len(clip.keys())
            for x in range(0, clip_len-(clip_len % iter_len), iter_len):
                new_train[counter] = { y:clip[y] for y in range(x,x+iter_len) }
                counter+=1
        self.train_dict = new_train

        # Validation
        new_valid = {}
        new_self_output = {}
        counter = 0
        for idx, clip in enumerate(valid_dict.values()):
            clip_len = len(clip.keys())
            new_self_output[idx] = clip
            for x in range(0, clip_len-(clip_len % iter_len), iter_len):
                new_valid[counter] = { y:clip[y] for y in range(x,x+iter_len) }
                counter+=1
        self.valid_dict = new_valid
        self.self_out_dict = new_self_output







    #############################################################
    # Saving dataset helper functions
    def arr_from_img(self, im,shift=0):
        w,h=im.size
        arr=im.getdata()
        c = int(np.product(arr.size) / (w*h))
        return np.asarray(arr, dtype=np.float32).reshape((h,w,c)).transpose(2,1,0) / 255.0 - shift
    
    def get_picture_array(self, X, index, shift=0):
        ch, w, h = X.shape[1], X.shape[2], X.shape[3]
        ret = ((X[index]+shift)*255.).reshape(ch,w,h).transpose(2,1,0).clip(0,255).astype(np.uint8)
        if ch == 1:
            ret=ret.reshape(h,w)
        return ret
    
    # loads mnist from web on demand
    def load_dataset(self):
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)
        import gzip
        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
            return data / np.float32(255)
        return load_mnist_images('train-images-idx3-ubyte.gz')
    
    # generates and returns video frames in uint8 array
    def generate_moving_mnist(self, shape, seq_len, seqss, num_sz, nums_per_images):
        """
        shape:          (int,int)   Shape of desired frames
        seq_len:        int         How many frames in sequences
        seqss:          [int,int..] How many gifs for each subset, ALIGN WITH nums_per_images
        num_sz:         int         Cant remember
        nums_per_images [int,int..] e.g. [2,3] = some videos with 2 MMNIST digits, and some with 3      
        """
        root_dir = "data/moving_mnist/various"
        root_dir = os.path.join(os.path.dirname(__file__), root_dir)
        #root_dir = self.args.dataset_path
        mnist = self.load_dataset()
        width, height = shape
        lims = (x_lim, y_lim) = width-num_sz, height-num_sz
        for xx, nums_per_image in enumerate(nums_per_images):
            seqs = seqss[xx]
            sub_root_dir = os.path.join(root_dir, str(nums_per_image))
            utils.mkdir_replc(sub_root_dir)
            dataset = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
            prcnt = round(seqs/100)
            prcnt_cnt = 0
            for seq_idx in range(seqs):
                if seq_idx%prcnt == 0:
                    print(f"{prcnt_cnt}% done!")
                    prcnt_cnt += 1
                sub_sub_root_dir = os.path.join(sub_root_dir, f"{seq_idx:05d}")
                # randomly generate direc/speed/position, calculate velocity vector
                utils.mkdir_replc(sub_sub_root_dir)
                direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
                speeds = np.random.randint(5, size=nums_per_image)+2
                veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in tuple(zip(direcs, speeds))]
                mnist_images = [Image.fromarray(self.get_picture_array(mnist,r,shift=0)).resize((num_sz,num_sz), Image.ANTIALIAS) \
                       for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
                positions = [(np.random.rand()*x_lim, np.random.rand()*y_lim) for _ in range(nums_per_image)]
                for frame_idx in range(seq_len):
                    canvases = [Image.new('L', (width,height)) for _ in range(nums_per_image)]
                    canvas = np.zeros((1,width,height), dtype=np.float32)
                    for i,canv in enumerate(canvases):
                        canv.paste(mnist_images[i], tuple(map(lambda p: int(round(p)), positions[i])))
                        canvas += self.arr_from_img(canv, shift=0)
                    # update positions based on velocity
                    next_pos = [tuple(map(sum, tuple(zip(p,v)))) for p,v in tuple(zip(positions, veloc))]
                    # bounce off wall if a we hit one
                    for i, pos in enumerate(next_pos):
                        for j, coord in enumerate(pos):
                            if coord < -2 or coord > lims[j]+2:
                                veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))
                    positions = [tuple(map(sum, tuple(zip(p,v)))) for p,v in tuple(zip(positions, veloc))]
                    # copy additive canvas to data array
                    imgsv = (canvas * 255).astype(np.uint8).clip(0,255)
                    imgsv = Image.fromarray(imgsv[0])
                    imgsv.save(os.path.join(sub_sub_root_dir, f"frame_{frame_idx:03d}.png"))
                    #save_image(torch.from_numpy(imgsv), os.path.join(sub_sub_root_dir, f"frame_{frame_idx:03d}.png"))
                    # Save frame

                    dataset[seq_idx*seq_len+frame_idx] = imgsv #(canvas * 255).astype(np.uint8).clip(0,255)
        sys.exit()
        return dataset
    
    def save_dataset(self, dest, filetype, frame_size, seq_len, seqs, num_sz, nums_per_image):
        dat = self.generate_moving_mnist((frame_size,frame_size), seq_len, seqs, num_sz, nums_per_image)
        dat = dat.reshape(seqs, seq_len, 1, frame_size, frame_size) # You will have to change the 1 for images with multiple channels
        n = seqs * seq_len
        if filetype == 'hdf5':
            import h5py
            raise Exception("Have not implemented for h5 extraction")
            from fuel.datasets.hdf5 import H5PYDataset
            def save_hd5py(dataset, destfile, indices_dict):
                f = h5py.File(destfile, mode='w')
                images = f.create_dataset('images', dataset.shape, dtype='uint8')
                images[...] = dataset
                split_dict = dict((k, {'images':v}) for k,v in indices_dict.iteritems())
                f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
                f.flush()
                f.close()
            indices_dict = {'train': (0, n*9/10), 'test': (n*9/10, n)}
            save_hd5py(dat, dest, indices_dict)
        elif filetype == 'npz':
            np.savez(dest, dat)
        elif filetype == 'jpg':
            for i in range(dat.shape[0]):
                Image.fromarray(self.get_picture_array(dat, i, shift=0)).save(os.path.join(dest, '{}.jpg'.format(i)))
    #############################################################




































#class VMDataset_v1(Dataset):
#    def __init__(self, args):
#        self.mode = "train"
#        self.args = args
#        if args.extract_dset:
#            return None
#        total_dset = utils.load_pickle( os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.dataset_path))
#
#        if args.shuffle:
#            random.shuffle(total_dset)
#        if args.dset_sze == -1: # If no specific dataset size is specified, use all of it
#            self.train_dict = { idx:data for idx, data in enumerate(total_dset[:round(len(total_dset)*args.train_ratio)]) }
#            self.valid_dict = { idx:data for idx, data in enumerate(total_dset[round(len(total_dset)*args.train_ratio):]) }
#        else:
#            val_size = round(args.dset_sze / args.train_ratio) - args.dset_sze  # Account for validation size too
#            if len(total_dset) < (val_size + args.dset_sze):
#                raise Exception(f"{val_size + args.dset_sze} needed samples for training ({args.dset_sze}) and validating ({val_size}) with specified train ratio of {args.train_ratio}. Exceeds available sample count: {len(total_dset)}")
#            else:
#                self.train_dict = { idx:data for idx, data in enumerate(total_dset[:args.dset_sze]) }
#                self.valid_dict = { idx:data for idx, data in enumerate(total_dset[args.dset_sze:(args.dset_sze+val_size)]) }
#        self.current_data_dict = self.train_dict
#        assert(args.in_no+args.out_no == len(self.current_data_dict[0]['frame_paths']),
#            "In frames + Ground truth frames do not equal the frame sample size of the dataset")
#        
#        # Sort Image reading method
#        img_read_method_switch = {
#            'binary'    : read_ims_binary,
#            'greyscale' : read_ims_greyscale,
#            'RGB'       : None
#        }
#        if args.img_type == "RGB":
#            raise Exception("RGB image reading not implemented")
#        self.img_read_method = img_read_method_switch[args.img_type]
#
#    def __len__(self):
#        return(len(self.current_data_dict))
#
#    def __getitem__(self, idx): # Indexs must count from 0
#        data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
#        positions, gt_positions = data['positions'][:self.args.in_no], data['positions'][self.args.in_no:]
#        frames = data['frame_paths']
#        frames = self.img_read_method(frames)
#        frames = torch.stack(frames, dim=0)
#        frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
#        return (frames, gt_frames)
#        #return (frames, positions, gt_frames, gt_positions) # Variably sized not solved yet
#
#    def set_mode(self, mode):
#        """
#        Jump between training/validation mode
#        """
#        self.mode = mode
#        if self.mode == 'train':
#            self.current_data_dict = self.train_dict
#        elif self.mode == 'valid':
#            self.current_data_dict = self.valid_dict
#
#    def save_dataset(self, in_no, out_no, save_path):
#        """
#        Save the dataset as preprocessed json
#        in_no = number of images for input
#        out_no = number of images for ground truth
#        """
#        self.frame_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), self.args.raw_frame_rootdir)
#        vids = os.listdir(self.frame_path)
#        try:
#            vids.remove('config.yml')
#        except:
#            pass
#        vids = [vid for vid in vids if not vid.endswith('.pickle')]
#        self.current_data_dict = [ os.path.join(self.frame_path, vid) for vid in vids ]
#        self.total_data ={ idx:path for idx, path in enumerate(self.current_data_dict) }
#        dataset_save = []
#        for vidx, vid_path in enumerate(self.total_data.values()):
#            print(vidx)
#            vid_name = vid_path.split('/')[-1]
#            try:
#                positions = utils.read_csv(os.path.join(vid_path, 'positions.csv'))
#            except FileNotFoundError:
#                print(f"{vid_path} not found.\nCreating dummy 'positions' information")
#                import pandas as pd
#                frame_cnt = len(os.listdir(vid_path))
#                positions = {
#                    'timestep': [i for i in range(frame_cnt)],
#                    'x': [0]*frame_cnt,
#                    'y': [0]*frame_cnt
#                }
#                positions = pd.DataFrame(data=positions)
#            indexs = torch.tensor(positions.values)[:,:1].long()
#            positions = torch.tensor(positions.values)[:,1:]    # Remove the useless frame index for now
#            frames = os.listdir(vid_path)
#            frames.sort()
#            try:
#                frames.remove('config.yml')
#            except:
#                pass
#            try: 
#                frames.remove('positions.csv')
#            except:
#                pass
#            try:    
#                frames.remove('simulation.gif')
#            except: pass
#            frames = [ os.path.join(vid_path, frame) for frame in frames ]
#            frame_paths = frames
#
#            for x in range(0, len(frames), in_no+out_no):
#                if(x+in_no+out_no<=len(frames)):    # Drop the remaining frames
#                    dataset_save.append(
#                        {
#                            "vid":vid_name,
#                            "vid_path":vid_path,
#                            "frame_idxs":indexs[x:x+in_no+out_no],
#                            "frame_paths":frame_paths[x:x+in_no+out_no],
#                            "positions":positions[x:x+in_no+out_no]
#                            #"frames":frames[x:x+in_no+out_no]
#                        }
#                    )
#        utils.save_pickle(dataset_save, os.path.join(save_path, str(in_no+out_no)+"_dset.pickle"))
#        # Frames, positions and indexs stacked, save a list of these
