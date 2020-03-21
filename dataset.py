__author__ = "Jumperkables"

import os, sys, random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

import utils

class VMDataset_v1(Dataset):
    def __init__(self, args):
        self.mode = "train"
        self.args = args
        total_dset = utils.load_pickle(args.dataset_path)
        if args.shuffle:
            random.shuffle(total_dset)
        self.train_dict = { idx:data for idx, data in enumerate(total_dset[:round(len(total_dset)*args.train_ratio)]) }
        self.valid_dict = { idx:data for idx, data in enumerate(total_dset[round(len(total_dset)*args.train_ratio):]) }
        self.current_data_dict = self.train_dict
        assert(args.in_no+args.out_no == len(self.current_data_dict[0]['frame_paths']),
            "In frames + Ground truth frames do not equal the frame sample size of the dataset")

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
        data = self.current_data_dict[idx]  #data.keys = ['vid', 'vid_path', 'frame_idxs', 'frame_paths', 'positions']
        positions, gt_positions = data['positions'][:self.args.in_no], data['positions'][self.args.in_no:]
        frames = [ ToTensor()(Image.open(frame)) for frame in data['frame_paths'] ]
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





    """
    For in case i need to do some raw frame loding again
    """
    def raw_init(self, args):
        """
        If you want to read the raw files
        """
        self.frame_path = os.path.expanduser("~/kable_management/data/visual_modelling/dataset_v1.0/raw")
        vids = os.listdir(self.frame_path)
        vids.remove('config.yml')
        self.current_data_dict = [ os.path.join(self.frame_path, vid) for vid in vids ]
        self.total_data ={ idx:path for idx, path in enumerate(self.current_data_dict) }
        #self.train_dict = { idx:path for idx, path in enumerate(self.current_data_dict[:round(len(self.current_data_dict)*args.train_ratio)]) }
        #self.valid_dict = { idx:path for idx, path in enumerate(self.current_data_dict[round(len(self.current_data_dict)*args.train_ratio):]) }

    def raw_getitem(self, idx): # Indexs must count from 0
        vid_path = self.current_data_dict[idx]
        positions = utils.read_csv(os.path.join(vid_path, 'positions.csv'))
        positions = torch.tensor(positions.values)[:,1:]    # Remove the useless frame index for now
        
        frames = os.listdir(vid_path)
        frames.sort()
        frames.remove('config.yml')
        frames.remove('positions.csv')
        frames.remove('simulation.gif')
        frames = [ os.path.join(vid_path, frame) for frame in frames ]
        frames = [ ToTensor()(Image.open(frame)) for frame in frames ]
        frames = torch.stack(frames, dim=0)
        pred_cutoff = round(0.9*len(frames))

    def save_dataset(self, in_no, out_no, save_path):
        """
        Save the dataset as preprocessed json
        in_no = number of images for input
        out_no = number of images for ground truth
        """
        self.raw_init(self.args)
        dataset_save = []
        for vidx, vid_path in enumerate(self.total_data.values()):
            print(vidx)
            vid_name = vid_path.split('/')[-1]
            positions = utils.read_csv(os.path.join(vid_path, 'positions.csv'))
            indexs = torch.tensor(positions.values)[:,:1].long()
            positions = torch.tensor(positions.values)[:,1:]    # Remove the useless frame index for now
            frames = os.listdir(vid_path)
            frames.sort()
            frames.remove('config.yml')
            frames.remove('positions.csv')
            frames.remove('simulation.gif')
            frames = [ os.path.join(vid_path, frame) for frame in frames ]
            frame_paths = frames
            frames = [ ToTensor()(Image.open(frame)) for frame in frames ]
            frames = torch.stack(frames, dim=0)

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
            
