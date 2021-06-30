__author__ = "Jumperkables, Daniel Kluvanec"

import os
import getpass
import cv2
import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

import tools.utils as utils


###########################################################################################
# Utility methods
###########################################################################################
# Image reading methods
def read_ims_binary(frames):
    raise Exception("Not implemented binary image read yet")


def read_ims_greyscale(frames):
    home_dir = os.path.expanduser("~").split("/")[1]
    frames_ret = []
    for frame in frames:
        frame = torch.from_numpy(cv2.imread(frame.replace('jumperkables', getpass.getuser()).replace("/home/", f"/{home_dir}/"), cv2.IMREAD_GRAYSCALE))
        if frame.shape != (64, 64):
            frame = F.interpolate(frame.unsqueeze(0).unsqueeze(0), (64, 64)).squeeze(0).squeeze(0)
        frames_ret.append(frame)
    return frames_ret


###########################################################################################
# Dataset Classes
###########################################################################################
class Simulations(Dataset):
    """
    root_dir -> clip_0,...clip_999 -> frame_00.png-frame_99.png AND positions.csv AND simulation.gif AND config.yml
    """
    def __init__(self, dataset_path, subset, mode, args, segmentation_flag=False, yaml_return=None):
        """
        :param dataset_path: str
        :param subset: 'train' or 'val'
        :param mode: 'consecutive' input is in_no frames, output is out_no frames which follow after input
                     'overlap' both input and output is in_no frames long, where the target sequence is shifted by out_no
                     'full_out' input is the first in_no frames, output is all the remaining frames in the sequence
        :param args: see VM_train.py
        :param segmentation_flag: bool
        :param yaml_return:
        """
        self.subset = subset
        self.mode = mode
        self.args = args
        # Flags
        self.segmentation_flag = segmentation_flag
        self.yaml_return = yaml_return  # To control what the yaml file should output for our simulations
        # Data processing
        vids_params = self.find_vids(dataset_path)
        train_vids_params, val_vids_params = self.train_val_split(vids_params, args.split_condition)
        if self.subset == 'train':
            vids_params = train_vids_params
        elif self.subset == 'val':
            vids_params = val_vids_params
        else:
            raise ValueError(f"Unknown subset {self.subset}")

        self.data_params = self.prepare_data(vids_params, mode, args.in_no, args.out_no)

        if args.img_type == 'binary':
            self.img_read_method = read_ims_binary
        elif args.img_type == 'greyscale':
            self.img_read_method = read_ims_greyscale
        elif args.img_type == 'RGB':
            raise NotImplementedError("RGB image reading not implemented")
        else:
            raise ValueError(f"Unknown img_type {args.img_type}")

    def __len__(self):
        return len(self.data_params)

    def __getitem__(self, idx):
        data_params = self.data_params[idx]
        data_paths = data_params['image_paths']

        if self.segmentation_flag:
            data_paths_in = data_paths[data_params['i_in_start']:data_params['i_in_end']]
            data_paths_out = data_paths[data_params['i_out_start']:data_params['i_out_end']]
            for i in range(len(data_paths_out)):
                head, tail = os.path.split(data_paths_out[i])
                data_paths_out[i] = os.path.join(head, 'mask', tail)
            frames_in = self.img_read_method(data_paths_in)
            frames_out = self.img_read_method(data_paths_out)
            frames_in = torch.stack(frames_in, dim=0)
            frames_out = torch.stack(frames_out, dim=0)
        else:
            frames = self.img_read_method(data_paths)
            frames = torch.stack(frames, dim=0)
            frames_in = frames[data_params['i_in_start']:data_params['i_in_end']]
            frames_out = frames[data_params['i_out_start']:data_params['i_out_end']]

        if self.yaml_return is None:
            yaml_return = 0
        elif self.yaml_return == "pendulum":    # Assuming pendulum predicts gravity
            yaml_return = [data_params['config']['SIM.GRAVITY']]
            yaml_return = torch.tensor(yaml_return).float()
        elif self.yaml_return == "bounces":
            yaml_return = [data_params['config']['bounces']['ball-ball'], data_params['config']['bounces']['wall']]
            yaml_return = torch.tensor(yaml_return).float()
        elif self.yaml_return == "grav":
            yaml_return = [data_params['config']['gy']]
            yaml_return = torch.tensor(yaml_return).float()
        elif self.yaml_return == "roller":
            yaml_return = [data_params['config']['SIM.GRAVITY']]
            yaml_return = torch.tensor(yaml_return).float()
        else:
            raise NotImplementedError(f"No yaml elements for {self.yaml_return} prepared for")

        return frames_in, frames_out, data_params['vid_name'], yaml_return

    @staticmethod
    def prepare_data(vids_params, mode, in_no, out_no):
        """
        Takes the vids params and creates input/output pairs using the correct self.mode
        :param vids_params: returned from self.find_vids
        :param mode: 'consecutive' input is in_no frames, output is out_no frames which follow after input
                     'overlap' both input and output is in_no frames long, where the target sequence is shifted by out_no. If in_no is None, then the full length of the video is used
                     'full_out' input is the first in_no frames, output is all the remaining frames in the sequence
        :param in_no: changes function based on mode (can be None if mode is overlap)
        :param out_no: changes function based on mode
        :returns: A list of datapoints, each being an dict of the format:
            {vid_name=str, config=dict, image_paths=[str], i_in_start=int, i_in_end=int, i_out_start=int, i_out_end=int}
        """
        dataset_params = []
        for vid_params in vids_params:
            vid_name = vid_params['vid_name']
            config = vid_params['config']
            image_paths = vid_params['image_paths']
            vid_length = len(image_paths)
            if mode == 'consecutive':
                segment_length = in_no + out_no
                i_in_end = in_no
                i_out_start = in_no
            elif mode == 'overlap':
                segment_length = vid_length
                i_in_end = vid_length - 1
                i_out_start = 1
            elif mode == 'full_out':
                segment_length = vid_length
                i_in_end = in_no
                i_out_start = in_no
            else:
                raise ValueError(f"Unknown mode: '{mode}'. Must be one of 'consecutive', 'overlap' or 'full_out'")

            for idx in range(0, vid_length, segment_length):
                if idx + segment_length > vid_length:
                    break
                data_params = {
                    'vid_name': vid_name,
                    'config': config,
                    'image_paths': image_paths[idx:idx + segment_length],
                    'i_in_start': 0,
                    'i_in_end': i_in_end,
                    'i_out_start': i_out_start,
                    'i_out_end': segment_length
                }
                dataset_params.append(data_params)
        return dataset_params

    @staticmethod
    def train_val_split(vids_params, condition):
        """
        Return train and validation data_params subsets
        """
        # See the argparse in main for a description of splitting functions
        if condition[:8] == "tv_ratio":
            tv_ratio = condition[9:].split('-')
            tv_ratio = float(tv_ratio[0])/(float(tv_ratio[0]) + float(tv_ratio[1]))

            # shuffle data_params in fixed manner
            rng = np.random.Generator(np.random.PCG64(2667))
            cutoff = round(len(vids_params) * tv_ratio)
            vids_params.sort(key=lambda x: x['image_paths'][0])
            rng.shuffle(vids_params)
            train_dict, val_dict = vids_params[:cutoff], vids_params[cutoff:]

            return train_dict, val_dict
        else:
            raise ValueError(f"Condition: {condition} not recognised")

    @staticmethod
    def find_vids(dataset_path):
        """
        Finds the directories and configs of all videos represented as directories of images
        This is done recursively through the dataset path. Any directories and subdirectories named 'mask' are ignored
        Returns:
            [{vid_name=str, config=dict, image_paths=[str]}]
        """
        dataset_path = os.path.abspath(dataset_path)
        # finds all subdirectories that contain
        vids_features = []
        for root, dirs, files in os.walk(dataset_path, topdown=True):
            if 'mask' in root.split('/'):
                continue
            if any(filename[-3:] in ('png', 'jpg') for filename in files):
                # root is now a subdirectory containing .png or .jpeg images

                # image params
                image_names = [filename for filename in files if filename[-3:] in ('png', 'jpg')]
                image_names.sort()
                image_paths = [os.path.join(root, image_name) for image_name in image_names]

                # config
                config_path = os.path.join(root, 'config.yml')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as config_file:
                        config = yaml.load(config_file, Loader=yaml.Loader)
                else:
                    print(f"CONFIG FILES DO NOT EXIST: {config_path}")
                    config = {}

                vid_features = {
                    'vid_name': os.path.basename(root),
                    'config': config,
                    'image_paths': image_paths
                }
                vids_features.append(vid_features)
        return vids_features
