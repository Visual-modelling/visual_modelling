__author__ = "Jumperkables"

import os, sys
sys.path.insert(1, os.path.expanduser("~/kable_management/blp_paper/tvqa/mystuff"))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset import VMDataset_v1
import utils

def train(args, dset, model):
    dset.set_mode("train")
    train_loader = DataLoader(dset, batch_size=args.bsz, shuffle=True)#, collate_fn=pad_collate)
    for batch_idx, batch in enumerate(train_loader):
        frames, positions, gt_frames, gt_positions = batch
        import ipdb; ipdb.set_trace()

def validate(args, dset, model):
    dset.set_mode("val")
    valid_loader = DataLoader(dset, batch_size=args.val_bsz, shuffle=True)#, collate_fn=pad_collate)
    for batch_idx, batch in enumerate(valid_loader):
        frames, positions, gt_frames, gt_positions = batch
        
if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    args = parser.parse_args()
    print(args)
    dset = VMDataset_v1(args)
    train(args, dset, None)
