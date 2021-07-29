import os, sys, copy
import argparse
import shutil

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from dataset import SimulationsPreloaded
from models.patch_transformer import VM_MixSeg

import tools.radam as radam
import tools.loss
import tools.utils as utils

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def load_pt(args):
    model = VM_MixSeg(in_chans=args.in_no, out_chans=args.out_no, img_size=64, return_attns=True)
    # Checkpointing
    if args.model_path != "":   # Empty string implies no loading
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        state_dict = {".".join(mod_name.split(".")[1:]):mod for mod_name, mod in state_dict.items()}
        state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys()} 
        # handle differences in inputs by repeating or removing input layers
        if args.in_no != state_dict['encoder.patch_embed1.proj.weight'].shape[1]:
            increase_ratio = math.ceil(args.in_no / state_dict['encoder.patch_embed1.proj.weight'].shape[1])
            state_dict['encoder.patch_embed1.proj.weight'] = state_dict['encoder.patch_embed1.proj.weight'].repeat(1,increase_ratio,1,1)[:,:args.in_no]
            increase_ratio = math.ceil(args.out_no / (state_dict["decode_head.linear_pred.weight"].shape[0]))
            state_dict["decode_head.linear_pred.weight"] = state_dict["decode_head.linear_pred.weight"].repeat(increase_ratio,1,1,1)[:16*args.out_no]
            state_dict["decode_head.linear_pred.bias"] = state_dict["decode_head.linear_pred.bias"].repeat(increase_ratio)[:16*args.out_no]
        # need to change the names of the state_dict keys from preloaded model
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Run specific arguments")
    parser.add_argument("--task", type=str, choices=["mnist","mocap","hdmb51","pendulum-regress","roller-regress","roller-pred","segmentation","bounces-regress","bounces-pred","grav-regress","grav-pred","moon-regress","blocks-regress"], help="Which task, classification or otherwise, to apply")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=1, help="minimum number of epochs to run.")
    parser.add_argument("--early_stopping", type=int, default=-1, help="number of epochs after no improvement before stopping, -1 to disable")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--n_imgs", type=int, default=20, help="Number of attention images to plot to wandb")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6, help="Setting default to what it was, it should likely be lower")
    parser.add_argument("--num_workers", type=int, default=0, help="Pytorch dataloader workers")
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--wandb", action="store_true", help="Save models/validation things to checkpoint location")
    parser.add_argument("--wandb_entity", type=str, default="visual-modelling", help="wandb entity to save project and run in")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")

    parser.add_argument_group("Dataset specific arguments")
    parser.add_argument("--split_condition", type=str, default="tv_ratio:8-1-1", help="Custom string deciding how to split datasets into train/test. Affiliated with a custom function in dataset")
    parser.add_argument("--dataset", type=str, nargs="+", choices=["mmnist", "simulations", "mocap", "hdmb51"], help="Type of dataset")
    parser.add_argument("--dataset_path", type=str, nargs="+", default=os.path.expanduser("~/"), help="Dataset paths")
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")

    parser.add_argument_group("Shared Model arguments")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "image_transformer", "PatchTrans"], help="Type of model to run")
    parser.add_argument("--model_path", type=str, default="", help="path of saved model")
    parser.add_argument("--linear_probes", action="store_true", help="Use linear probes as output instead")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--loss", type=str, default="MSE", choices=["mse", "sl1", "focal", "ssim", "mnist"], help="Loss function for the network")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")

    parser.add_argument_group("2D and 3D CNN specific arguments")
    parser.add_argument("--encoder_freeze", action="store_true", help="freeze the CNN/transformer layers and only train the linear cls layer afterwards")
    parser.add_argument("--krnl_size", type=int, default=3, help="Height and width kernel size")
    parser.add_argument("--krnl_size_t", type=int, default=3, help="Temporal kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Height and width Padding")
    parser.add_argument("--padding_t", type=int, default=1, help="Temporal Padding")
    parser.add_argument("--depth", type=int, default=2, help="depth of the updown")
    parser.add_argument("--channel_factor", type=int, default=64, help="channel scale factor for up down network")

    parser.add_argument_group("Transformer model specific arguments")
    parser.add_argument("--d_model", type=int, default=4096, help="The number of features in the input (flattened image dimensions)")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers to use")
    parser.add_argument("--nhead", type=int, default=8, help="The number of heads in the multiheadattention models")
    parser.add_argument("--dim_feedforward", type=int, default=16384, help="The dimension of the linear layers after each attention")
    parser.add_argument("--dropout", type=float, default=0.1, help="The dropout value")
    parser.add_argument("--pixel_regression_layers", type=int, default=1, help="How many layers to add after transformers")
    parser.add_argument("--norm_layer", type=str, default="layer_norm", choices=["layer_norm", "batch_norm"], help="What normalisation layer to use")
    parser.add_argument("--output_activation", type=str, default="linear", choices=["linear-256", "hardsigmoid-256", "sigmoid-256"], help="What activation function to use at the end of the network")
    parser.add_argument("--pos_encoder", type=str, default="add", help="What positional encoding to use. 'none', 'add', 'add_runtime', or an integer concatenation with the number of bits to concatenate.")
    parser.add_argument("--mask", action="store_true", help="Whether to add a triangular attn_mask to the transformer attention")

    args = parser.parse_args()
    print(args)

    ########################################################################################################################
    ######## ERROR CONDITIONS To make sure erroneous runs aren't accidentally executed
    if not args.task == "mnist":
        assert len(args.dataset) == len(args.dataset_path), f"Number of specified dataset paths and dataset types should be equal"

    if args.task in ["hdmb51","mocap"]:
        raise NotImplementedError("Haven't reimplemented this yet. May not be worth it in the end.")

    if args.task == "segmentation":
        assert args.out_no == 1, f"Segmentation is only well defined with out_no == 1. in_no is handled separately"

    assert args.model == "PatchTrans", "This script is for the patch transformers only"
    ########################################################################################################################
    ########################################################################################################################

    # Create full model path
    if args.model_path != "":
        args.model_path = os.path.join(os.path.dirname(__file__), args.model_path)

    # GPU
    if args.device == -1:
        gpus = None
    else: 
        gpus = [args.device]  # TODO Implement multi GPU support

    # Logging and Saving: If we're saving this run, prepare the neccesary directory for saving things
    wandb.init(entity=args.wandb_entity, project="visual-modelling", name=args.jobname)
    wandb_logger = pl.loggers.WandbLogger(offline=not args.wandb)#, resume="allow")
    wandb_logger.log_hyperparams(args)
    repo_rootdir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_dir = os.path.join(repo_rootdir, ".results", args.jobname )
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    args.results_dir = results_dir

    ################################
    # MMNIST
    ################################
    if args.task == "mnist":
        """
        You don't need to set a dataset or dataset path for MNIST. Its all handled here since its so small and easy to load
       """
        train_dset = MNIST(train=True, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
        valid_test_dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
        valid_dset = torch.utils.data.Subset(valid_test_dset, list(range(0, len(valid_test_dset)//2)))
        test_dset = torch.utils.data.Subset(valid_test_dset, list(range(len(valid_test_dset)//2, len(valid_test_dset))))

    ################################
    # Segmentation
    ################################
    elif args.task == "segmentation":
        """
        Point the dataset to the root directory
        """
        copy_args = copy.deepcopy(args)
        copy_args.in_no = 1
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', copy_args, segmentation_flag=True)
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    ################################
    # Roller regression/prediction
    ################################   
    elif args.task in ["roller-regress","roller-pred"]:
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="roller")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    ################################
    # Moon regression/prediction
    ################################   
    elif args.task in ["moon-regress"]:
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="moon")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    ################################
    # Block mass ratio regression
    ################################   
    elif args.task in ["blocks-regress"]:
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="blocks")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    ################################
    # Pendulum
    ################################   
    elif args.task == "pendulum-regress":
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="pendulum")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    ################################
    # Gravity regression/prediction
    ################################   
    elif args.task in ["grav-regress","grav-pred"]:
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="grav")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    ################################
    # Ball bounces regression/prediction
    ################################   
    elif args.task in ["bounces-regress","bounces-pred"]:
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="bounces")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')

    del train_dset
    del valid_dset
    test_loader = DataLoader(test_dset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=False)
    test_loader = iter(test_loader)
    model = load_pt(args)
    model = model.encoder
    model.to(args.device)

    with tqdm(total=args.n_imgs) as pbar:
        for nimg in range(args.n_imgs):
            pbar.update(1)
            pbar.set_description(f"Self output: {nimg+1}/{args.n_imgs}")
            start_frames, gt_frames, vid_name, _ = next(test_loader)
            start_frames = start_frames.float().to(args.device)
            outs, logits_ret = model(start_frames)
            # r b g y
            small_box = logits_ret[0].mean(dim=1)[0].argmax().cpu().detach()
            bigger_box = logits_ret[1].mean(dim=1)[0].argmax().cpu().detach()
            bigga_box = logits_ret[2].mean(dim=1)[0].argmax().cpu().detach()
            biggest_box = logits_ret[3].mean(dim=1)[0].argmax().cpu().detach()
            plot_frame = start_frames.mean(dim=1).permute(1,2,0).cpu().detach().numpy()
            fig, ax = plt.subplots()
            ax.imshow(plot_frame, cmap="gray", vmax=255, vmin=0)
            corner = {
                0:(0,0),
                1:(32,0),
                2:(0,32),
                3:(32,32)
            }
            rect = patches.Rectangle(corner[int(small_box)], 32, 32, linewidth=1, edgecolor='r', facecolor='none')
            rect = patches.Rectangle(corner[int(bigger_box)], 32, 32, linewidth=1, edgecolor='b', facecolor='none')
            rect = patches.Rectangle(corner[int(bigga_box)], 32, 32, linewidth=1, edgecolor='g', facecolor='none')
            rect = patches.Rectangle(corner[int(biggest_box)], 32, 32, linewidth=1, edgecolor='y', facecolor='none')
            ax.add_patch(rect)
            plt.show()
            print("Here")
