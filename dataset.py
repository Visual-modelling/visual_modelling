__author__ = "Jumperkables"

import os, sys, random
import getpass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import cv2

import tools.utils as utils

class VMDataset_v1(Dataset):
    def __init__(self, args):
        self.mode = "train"
        self.args = args
        if not args.extract_n_dset_file:
            total_dset = utils.load_pickle(args.dataset_path)
            if args.shuffle:
                random.shuffle(total_dset)
            self.train_dict = { idx:data for idx, data in enumerate(total_dset[:round(len(total_dset)*args.train_ratio)]) }
            self.valid_dict = { idx:data for idx, data in enumerate(total_dset[round(len(total_dset)*args.train_ratio):]) }
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

        return (frames, positions, gt_frames, gt_positions)

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
        self.frame_path = self.args.raw_frame_rootdir
        vids = os.listdir(self.frame_path)
        vids.remove('config.yml')
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
    frames = [ torch.from_numpy(cv2.imread(frame, cv2.IMREAD_GRAYSCALE)) for frame in frames]
    #old pillow version frames = [ (ToTensor()(Image.open(frame.replace('jumperkables', getpass.getuser()) ))>0).float() for frame in frames ]
    return frames
           
