__author__ = "Jumperkables"

import os, sys, argparse
import torch
from torch.utils.data import DataLoader

from dataset import VMDataset_v1
from models.FC3D import FC3D_1_0
import utils
import radam 
import loss

def train(args, dset, model, optimizer, criterion):
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=args.bsz, shuffle=True)
    for batch_idx, batch in enumerate(train_loader):
        import ipdb; ipdb.set_trace()
        frames, positions, gt_frames, gt_positions = batch
        frames = frames.squeeze(2).float()
        gt_frames = gt_frames.squeeze(2).float()
        out = model(frames).squeeze(2)
        loss = criterion(out, gt_frames)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(args, dset, model):
    dset.set_mode("val")
    model.valid()
    valid_loader = DataLoader(dset, batch_size=args.val_bsz, shuffle=True)
    for batch_idx, batch in enumerate(valid_loader):
        frames, positions, gt_frames, gt_positions = batch
        frames = frames.squeeze(2).float()
        gt_frames = gt_frames.squeeze(2).float()
        img = model(frames)

if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--raw_frame_rootdir", type=str, default=os.path.expanduser("~/"), help="The root directory containing 00000, 00001...")
    parser.add_argument("--extract_n_dset_file", action="store_true", help="activate this if you would like to extract your n_dset")
    parser.add_argument("--extracted_n_dset_savepathdir", type=str, default=os.path.expanduser("~/"), help="root directory of where you would like to save your n_dset.pickle")
    parser.add_argument("--dataset_path", type=str, default=os.path.expanduser("~/"))
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")
    """
    Please note, in_no + out_no must equal the size your dataloader is returning. Default is batches of 6 frames, 5 forward and 1 for ground truth
    """
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    args = parser.parse_args()
    print(args)
    dset = VMDataset_v1(args)
    model = FC3D_1_0(args)
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad],
                                    lr=3e-4, weight_decay=1e-5)
    criterion = loss.l2loss
    if args.extract_n_dset_file:
        dset.save_dataset(args.in_no, args.out_no, args.extracted_n_dset_savepathdir)
        print("Extraction successful. Saved to:\n", args.extracted_n_dset_savepathdir+"/"+str(args.in_no+args.out_no)+"_dset.pickle")
        sys.exit()
    train(args, dset, model, optimizer)