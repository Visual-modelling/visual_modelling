__author__ = "Jumperkables"

import os, sys

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
        self.frame_path = os.path.expanduser("~/kable_management/data/visual_modelling/dataset_v1.0")
        self.mode = None
        vids = os.listdir(self.frame_path)
        vids.remove('config.yml')
        self.current_data_dict = [ os.path.join(self.frame_path, vid) for vid in vids ]
        self.train_dict = { idx:path for idx, path in enumerate(self.current_data_dict[:round(len(self.current_data_dict)*args.train_ratio)]) }
        self.valid_dict = { idx:path for idx, path in enumerate(self.current_data_dict[round(len(self.current_data_dict)*args.train_ratio):]) }

    def __len__(self):
        return(len(self.current_data_dict))

    def __getitem__(self, idx): # Indexs must count from 0
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

        return (frames[:pred_cutoff], positions[:pred_cutoff], frames[pred_cutoff:], positions[pred_cutoff:])

    def set_mode(self, mode):
        """
        Jump between training/validation mode
        """
        self.mode = mode
        if self.mode == 'train':
            self.current_data_dict = self.train_dict
        elif self.mode == 'valid':
            self.current_data_dict = self.valid_dict