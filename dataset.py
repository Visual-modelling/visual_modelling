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

import tools.utils as utils

class Old_MMNIST(Dataset):
    def __init__(self, args):
        self.mode = "train"
        self.args = args
        total_dset = np.load( os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.dataset_path))
        total_dset = np.transpose(total_dset, (1,0,2,3))

        if args.shuffle:
            random.shuffle(total_dset)
        if args.dset_sze == -1: # If no specific dataset size is specified, use all of it
            self.train_dict = { idx:data for idx, data in enumerate(total_dset[:round(len(total_dset)*args.train_ratio)]) }
            self.valid_dict = { idx:data for idx, data in enumerate(total_dset[round(len(total_dset)*args.train_ratio):]) }
        else:
            val_size = round(args.dset_sze / args.train_ratio) - args.dset_sze  # Account for validation size too
            if len(total_dset) < (val_size + args.dset_sze):
                raise Exception(f"{val_size + args.dset_sze} needed samples for training ({args.dset_sze}) and validating ({val_size}) with specified train ratio of {args.train_ratio}. Exceeds available sample count: {len(total_dset)}")
            else:
                self.train_dict = { idx:data for idx, data in enumerate(total_dset[:args.dset_sze]) }
                self.valid_dict = { idx:data for idx, data in enumerate(total_dset[args.dset_sze:(args.dset_sze+val_size)]) }
                import ipdb; ipdb.set_trace()
                print("dsadsa")


        self.current_data_dict = self.train_dict
        
        # Sort Image reading method
        #img_read_method_switch = {
        #    'binary'    : read_ims_binary,
        #    'greyscale' : read_ims_greyscale,
        #    'RGB'       : None
        #}
        #if args.img_type == "RGB":
        #    raise Exception("RGB image reading not implemented")
        #self.img_read_method = img_read_method_switch[args.img_type]

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
        frames = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
        frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]

        return (frames, gt_frames)

    def set_mode(self, mode):
        """
        Jump between training/validation mode
        """
        self.mode = mode
        if self.mode == 'train':
            self.current_data_dict = self.train_dict
        elif self.mode == 'valid':
            self.current_data_dict = self.valid_dict







###########################################################################################
# Code adapted from: https://gist.github.com/tencia/afb129122a64bde3bd0c

# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################
class MMNIST(Dataset):
    def __init__(self, args):
        self.mode = "train"
        self.args = args
        if args.extract_dset:
            return None
        total_dset = np.load( os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.dataset_path))
        total_dset = total_dset['arr_0']
        if args.shuffle:
            random.shuffle(total_dset)
        if args.dset_sze == -1: # If no specific dataset size is specified, use all of it
            self.train_dict = { idx:data for idx, data in enumerate(total_dset[:round(len(total_dset)*args.train_ratio)]) }
            self.valid_dict = { idx:data for idx, data in enumerate(total_dset[round(len(total_dset)*args.train_ratio):]) }
        else:
            val_size = round(args.dset_sze / args.train_ratio) - args.dset_sze  # Account for validation size too
            if len(total_dset) < (val_size + args.dset_sze):
                raise Exception(f"{val_size + args.dset_sze} needed samples for training ({args.dset_sze}) and validating ({val_size}) with specified train ratio of {args.train_ratio}. Exceeds available sample count: {len(total_dset)}")
            else:
                self.train_dict = { idx:data for idx, data in enumerate(total_dset[:args.dset_sze]) }
                self.valid_dict = { idx:data for idx, data in enumerate(total_dset[args.dset_sze:(args.dset_sze+val_size)]) }
        self.current_data_dict = self.train_dict
        
        # Sort Image reading method
        #img_read_method_switch = {
        #    'binary'    : read_ims_binary,
        #    'greyscale' : read_ims_greyscale,
        #    'RGB'       : None
        #}
        #if args.img_type == "RGB":
        #    raise Exception("RGB image reading not implemented")
        #self.img_read_method = img_read_method_switch[args.img_type]

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
    def generate_moving_mnist(self, shape, seq_len, seqs, num_sz, nums_per_image):
        mnist = self.load_dataset()
        width, height = shape
        lims = (x_lim, y_lim) = width-num_sz, height-num_sz
        dataset = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
        prcnt = round(seqs/100)
        prcnt_cnt = 0
        for seq_idx in range(seqs):
            if seq_idx%prcnt == 0:
                print(f"{prcnt_cnt}% done!")
                prcnt_cnt += 1
            # randomly generate direc/speed/position, calculate velocity vector
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
                dataset[seq_idx*seq_len+frame_idx] = (canvas * 255).astype(np.uint8).clip(0,255)
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
 




class VMDataset_v1(Dataset):
    def __init__(self, args):
        self.mode = "train"
        self.args = args
        if args.extract_dset:
            return None
        total_dset = utils.load_pickle( os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.dataset_path))

        if args.shuffle:
            random.shuffle(total_dset)
        if args.dset_sze == -1: # If no specific dataset size is specified, use all of it
            self.train_dict = { idx:data for idx, data in enumerate(total_dset[:round(len(total_dset)*args.train_ratio)]) }
            self.valid_dict = { idx:data for idx, data in enumerate(total_dset[round(len(total_dset)*args.train_ratio):]) }
        else:
            val_size = round(args.dset_sze / args.train_ratio) - args.dset_sze  # Account for validation size too
            if len(total_dset) < (val_size + args.dset_sze):
                raise Exception(f"{val_size + args.dset_sze} needed samples for training ({args.dset_sze}) and validating ({val_size}) with specified train ratio of {args.train_ratio}. Exceeds available sample count: {len(total_dset)}")
            else:
                self.train_dict = { idx:data for idx, data in enumerate(total_dset[:args.dset_sze]) }
                self.valid_dict = { idx:data for idx, data in enumerate(total_dset[args.dset_sze:(args.dset_sze+val_size)]) }
        self.current_data_dict = self.train_dict
        assert(args.in_no+args.out_no == len(self.current_data_dict[0]['frame_paths']),
            "In frames + Ground truth frames do not equal the frame sample size of the dataset")
        
        # Sort Image reading method
        img_read_method_switch = {
            'binary'    : read_ims_binary,
            'greyscale' : read_ims_greyscale,
            'RGB'       : None
        }
        if args.img_type == "RGB":
            raise Exception("RGB image reading not implemented")
        self.img_read_method = img_read_method_switch[args.img_type]

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
        data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
        positions, gt_positions = data['positions'][:self.args.in_no], data['positions'][self.args.in_no:]
        frames = data['frame_paths']
        frames = self.img_read_method(frames)
        frames = torch.stack(frames, dim=0)
        frames, gt_frames = frames[:self.args.in_no], frames[self.args.in_no:]
        return (frames, gt_frames)
        #return (frames, positions, gt_frames, gt_positions) # Variably sized not solved yet

    def set_mode(self, mode):
        """
        Jump between training/validation mode
        """
        self.mode = mode
        if self.mode == 'train':
            self.current_data_dict = self.train_dict
        elif self.mode == 'valid':
            self.current_data_dict = self.valid_dict

    def save_dataset(self, in_no, out_no, save_path):
        """
        Save the dataset as preprocessed json
        in_no = number of images for input
        out_no = number of images for ground truth
        """
        self.frame_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), self.args.raw_frame_rootdir)
        vids = os.listdir(self.frame_path)
        vids.remove('config.yml')
        vids = [vid for vid in vids if not vid.endswith('.pickle')]
        self.current_data_dict = [ os.path.join(self.frame_path, vid) for vid in vids ]
        self.total_data ={ idx:path for idx, path in enumerate(self.current_data_dict) }
        dataset_save = []
        for vidx, vid_path in enumerate(self.total_data.values()):
            print(vidx)
            vid_name = vid_path.split('/')[-1]
            positions = utils.read_csv(os.path.join(vid_path, 'positions.csv'))
            #viddle_path = '/home/jumperkables/kable_management/data/visual_modelling/hudsons_og/2000/%.5d' % vidx
            #positions = utils.read_csv(os.path.join(viddle_path, 'positions.csv')) # For use if Tom's Positions are buggy

            indexs = torch.tensor(positions.values)[:,:1].long()
            positions = torch.tensor(positions.values)[:,1:]    # Remove the useless frame index for now
            frames = os.listdir(vid_path)
            frames.sort()
            frames.remove('config.yml')
            frames.remove('positions.csv')
            frames.remove('simulation.gif')
            frames = [ os.path.join(vid_path, frame) for frame in frames ]
            frame_paths = frames

            for x in range(0, len(frames), in_no+out_no):
                if(x+in_no+out_no<=len(frames)):    # Drop the remaining frames
                    dataset_save.append(
                        {
                            "vid":vid_name,
                            "vid_path":vid_path,
                            "frame_idxs":indexs[x:x+in_no+out_no],
                            "frame_paths":frame_paths[x:x+in_no+out_no],
                            "positions":positions[x:x+in_no+out_no]
                            #"frames":frames[x:x+in_no+out_no]
                        }
                    )
        utils.save_pickle(dataset_save, os.path.join(save_path, str(in_no+out_no)+"_dset.pickle"))
        # Frames, positions and indexs stacked, save a list of these
 





# Image reading methods
def read_ims_binary(frames):
    raise Exception("Not implemented binary image read yet")
    return frames
    #old pillow version return([ (ToTensor()(Image.open(frame.replace('jumperkables', getpass.getuser()) ))>0).float() for frame in frames ])

def read_ims_greyscale(frames):
    home_dir = os.path.expanduser("~").split("/")[1]
    frames = [ torch.from_numpy(cv2.imread(frame.replace('jumperkables', getpass.getuser()).replace("/home/", f"/{home_dir}/" ), cv2.IMREAD_GRAYSCALE)) for frame in frames]
    #old pillow version frames = [ (ToTensor()(Image.open(frame.replace('jumperkables', getpass.getuser()) ))>0).float() for frame in frames ]
    return frames
          

if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_dset", action="store_true", help="activate this if you would like to extract your n_dset")
    parser.add_argument("--extracted_dset_savepathdir", type=str, default=os.path.expanduser("~/"), help="root directory of where you would like to save your n_dset.pickle")
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--raw_frame_rootdir", type=str, default=os.path.expanduser("~/"), help="The root directory containing 00000, 00001...")
    parser.add_argument("--dataset", type=str, default="hudsons", choices=["hudsons","mmnist"])

    # MMNIST Specific settings
    parser.add_argument('--dest', type=str, dest='dest')
    parser.add_argument('--filetype', default='npz', type=str, dest='filetype')
    parser.add_argument('--frame_size', default=64, type=int, dest='frame_size')
    parser.add_argument('--seq_len', default=30, type=int, dest='seq_len') # length of each sequence
    parser.add_argument('--seqs', default=100, type=int, dest='seqs') # number of sequences to generate
    parser.add_argument('--num_sz', default=28, type=int, dest='num_sz') # size of mnist digit within frame
    parser.add_argument('--nums_per_image', default=2, type=int, dest='nums_per_image') # number of digits in each frame
    args = parser.parse_args()
    print(args)
    if not args.extract_dset:
        print("Must explicitly set --extract_dset flag to avoid accidental dataset erasure.")
        sys.exit()
    if args.dataset == "hudsons":
        dset = VMDataset_v1(args)
        dset.save_dataset(args.in_no, args.out_no, os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.extracted_dset_savepathdir))
        print("Extraction successful. Saved to:\n", args.extracted_dset_savepathdir+"/"+str(args.in_no+args.out_no)+"_dset.pickle")
    elif args.dataset == "mmnist":
        print("Finish implementing MMNIST")
        dset = MMNIST(args)
        dset.save_dataset(dest=args.dest, filetype=args.filetype, frame_size=args.frame_size, seq_len=args.seq_len, seqs=args.seqs, num_sz=args.num_sz, nums_per_image=args.nums_per_image)

    else:
        raise Exception(f"{args.dataset} dataset not implemented")
