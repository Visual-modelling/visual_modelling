import os, sys, copy
import argparse
import shutil
import math

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from dataset import SimulationsPreloaded, Simulations
from models.UpDown2D import FCUpDown2D
from models.transformer import ImageTransformer
import tools.radam as radam
import tools.loss
import tools.utils as utils

import numpy as np
import tqdm
from tqdm import tqdm
from time import sleep
import pandas as pd
import wandb



################################################################################################
# Pytorch lightning trainers
################################################################################################
class FcUpDown2D2Scalars(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model == 'UpDown2D':
            self.model = FCUpDown2D(args)

            # Checkpointing
            if args.model_path != "":   # Empty string implies no loading
                checkpoint = torch.load(args.model_path)
                state_dict = checkpoint['state_dict']
                state_dict = {".".join(mod_name.split(".")[1:]):mod for mod_name, mod in state_dict.items()}
                # handle differences in inputs by repeating or removing input layers
                if args.in_no != state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1]:
                    increase_ratio = math.ceil(args.in_no / state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1])
                    state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'] = state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].repeat(1,increase_ratio,1,1)[:,:args.in_no]
                    # Figure out the position of the final layer with depth
                    final_layer = (2*args.depth)+2-1    # -1 for python indexing
                    increase_ratio = math.ceil(args.out_no / state_dict[f"UDChain.layers.{final_layer}.conv.weight"].shape[0])
                    state_dict[f"UDChain.layers.{final_layer}.conv.weight"] = state_dict[f"UDChain.layers.{final_layer}.conv.weight"].repeat(increase_ratio,1,1,1)[:args.out_no]
                    state_dict[f"UDChain.layers.{final_layer}.conv.bias"] = state_dict[f"UDChain.layers.{final_layer}.conv.bias"].repeat(increase_ratio)[:args.out_no]
                # need to change the names of the state_dict keys from preloaded model
                self.model.load_state_dict(state_dict)
        elif args.model == 'image_transformer':
            self.model = ImageTransformer(args)
            if args.model_path != "":  # Empty string implies no loading
                checkpoint = torch.load(args.model_path)
                state_dict = checkpoint['state_dict']
                for old_key in list(state_dict.keys()):
                    state_dict[old_key[6:]] = state_dict.pop(old_key)
                state_dict['pos_encoder.pos_encoding'] = self.model.state_dict()['pos_encoder.pos_encoding']
                self.model.load_state_dict(state_dict)
        elif args.model == 'PatchTrans':
            from models.patch_transformer import VM_MixSeg
            self.model = VM_MixSeg(args=args, in_chans=args.in_no, out_chans=args.out_no, img_size=64)
            # Checkpointing
            if args.model_path != "":   # Empty string implies no loading
                checkpoint = torch.load(args.model_path)
                state_dict = checkpoint['state_dict']
                state_dict = {".".join(mod_name.split(".")[1:]):mod for mod_name, mod in state_dict.items()}
                # handle differences in inputs by repeating or removing input layers
                if args.in_no != state_dict['encoder.patch_embed1.proj.weight'].shape[1]:
                    increase_ratio = math.ceil(args.in_no / state_dict['encoder.patch_embed1.proj.weight'].shape[1])
                    state_dict['encoder.patch_embed1.proj.weight'] = state_dict['encoder.patch_embed1.proj.weight'].repeat(1,increase_ratio,1,1)[:,:args.in_no]
                    increase_ratio = math.ceil(args.out_no / (state_dict["decode_head.linear_pred.weight"].shape[0]))
                    state_dict["decode_head.linear_pred.weight"] = state_dict["decode_head.linear_pred.weight"].repeat(increase_ratio,1,1,1)[:16*args.out_no]
                    state_dict["decode_head.linear_pred.bias"] = state_dict["decode_head.linear_pred.bias"].repeat(increase_ratio)[:16*args.out_no]
                # need to change the names of the state_dict keys from preloaded model
                self.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # If we are loading a checkpoint, the we are freezing the rest of the pretrained layers

        # Freeze the CNN weights
        if args.encoder_freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # Different tasks will be expecting different output numbers
        if args.task == "mnist":
            n_outputs = 10
        elif args.task == "pendulum-regress":
            n_outputs = 1
        elif args.task == "roller-regress":
            n_outputs = 1
        elif args.task == "moon-regress":
            n_outputs = 1
        elif args.task == "blocks-regress":
            n_outputs = 1
        elif args.task == "roller-pred":
            n_outputs = 201 # [0,0.5,1.0,....,99.5,100]
        elif args.task in ["3dbounces-regress","2dbounces-regress"]:
            n_outputs = 1
        elif args.task == "grav-regress":
            n_outputs = 1
        elif args.task == "grav-pred":
            n_outputs = 7 # Prediction for gy for 2D bouncing dataset
        else:
            raise NotImplementedError("Task has not been implemented yet")

        # Probe classifier
        if args.model == 'UpDown2D':
            probe_len = (args.out_no*64*64) # Final layer
            for i in range(args.depth):
                probe_len +=  2 * args.channel_factor * (2**i)  * (64/(2**(i+1))) * (64/(2**(i+1)))  # once on the way up and back down
            probe_len += args.channel_factor * (2**args.depth) * (64/(2**(args.depth+1))) * (64/(2**(args.depth+1)))
        elif args.model == 'image_transformer':
            dummy_input = torch.zeros((1, args.in_no, 64, 64), device=self.device)
            _, hidden_xs = self.model(dummy_input)
            probe_len = sum(torch.numel(hidden_x) for hidden_x in hidden_xs)
            del dummy_input, hidden_xs
        elif args.model == 'PatchTrans':
            # From encoder
            probe_len = 64*16*16
            probe_len += 128*8*8
            probe_len += 256*4*4
            probe_len += 512*2*2
            # From decoder
            probe_len += 256*16*16
            probe_len += 256*16*16
            probe_len += 256*16*16
            probe_len += 256*16*16
            probe_len += 16*16*16


            # Outputs
            probe_len += 64*64
            
        else:
            raise ValueError(f"Unknown model: {args.model}")

        if self.args.linear_probes:
            self.probe_fc = nn.Linear(int(probe_len), n_outputs)
        else:
            self.probe_fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.args.out_no*64*64, 100),
                nn.BatchNorm1d(100),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(100, n_outputs)
            )

        # Validation metrics
        self.valid_acc = torchmetrics.Accuracy()

        # Training metrics
        self.train_acc = torchmetrics.Accuracy()

        # Test metrics
        self.test_acc = torchmetrics.Accuracy()

        if args.task in ["mnist","grav-pred","bounces-pred","roller-pred"]:
            self.criterion = nn.CrossEntropyLoss()
        elif args.task in ["pendulum-regress","3dbounces-regress","2dbounces-regress","grav-regress","roller-regress","moon-regress","blocks-regress"]:
            self.criterion = nn.SmoothL1Loss(beta=0.01)
        else:
            raise NotImplementedError(f"Task: '{args.task}' has not got a specified criterion")

    def configure_optimizers(self):
        optimizer = radam.RAdam([p for p in self.model.parameters()], lr=self.args.lr)
        return optimizer

    def forward(self, x):
        # Through the encoder
        if self.args.linear_probes:
            _, probe_ret = self.model(x)
            probe_ret = torch.cat([ tens.contiguous().view(x.shape[0], -1) for tens in probe_ret], dim=1)
        else:
            probe_ret, _ = self.model(x)
            probe_ret = probe_ret.view(probe_ret.shape[0], -1)
        # And then the classifier
        probe_ret = self.probe_fc(probe_ret)
        return probe_ret

    def training_step(self, train_batch, batch_idx):
        self.model.eval()
        if self.args.task == "mnist":
            frame, label = train_batch
            frames = frame.repeat(1,self.args.in_no,1,1)
        else:
            frames, _, _, label = train_batch
        frames = frames.float()
        out = self(frames)
        if self.args.task == "mnist":
            out = F.softmax(out, dim=1)
        train_loss = self.criterion(out, label)
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.args.task in ["mnist","mocap","hdmb51","grav-pred","bounces-pred","roller-pred"]:
            self.log("train_acc", self.train_acc(out, label), prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, valid_batch, batch_idx):
        if self.args.task == "mnist":
            frame, label = valid_batch
            frames = frame.repeat(1,self.args.in_no,1,1)
        else:
            frames, _, _, label = valid_batch
        frames = frames.float()
        out = self(frames)
        if self.args.task == "mnist":
            out = F.softmax(out, dim=1)
        valid_loss = self.criterion(out, label)
        if self.testing:    # TODO refine this. this is a quick workaround
            self.log("test_loss", valid_loss, on_step=False, on_epoch=True)
        else:
            self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
        if self.args.task in ["mnist","mocap","hdmb51","grav-pred","bounces-pred","roller-pred"]:
            if self.testing:
                self.log("test_acc", self.test_acc(out, label), prog_bar=True, on_step=False, on_epoch=True)
            else:
                self.log("valid_acc", self.valid_acc(out, label), prog_bar=True, on_step=False, on_epoch=True)
        else:
            if self.testing:
                self.log("test_l1", F.l1_loss(out, label), prog_bar=True, on_step=False, on_epoch=True)
            else:
                self.log("valid_l1", F.l1_loss(out, label), prog_bar=True, on_step=False, on_epoch=True)


    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)


#class FCUpDown2D_2_Segmentation(pl.LightningModule):
#    def __init__(self, args):
#        super().__init__()
#
#        self.args = args
#        self.model = FCUpDown2D(args)
#
#        # Freeze the CNN weights
#        if args.encoder_freeze:
#            raise ValueError("encoder_freeze argument does not make sense in this segmentation model")
#            for param in self.model.parameters():
#                param.requires_grad = False
#
#        # Checkpointing
#        if args.model_path != "":   # Empty string implies no loading
#            checkpoint = torch.load(args.model_path)
#            state_dict = checkpoint['state_dict']
#            # need to change the names of the state_dict keys from preloaded model
#            state_dict = {".".join(mod_name.split(".")[1:]):mod for mod_name, mod in state_dict.items()}
#            # handle differences in inputs by repeating or removing input layers
#            if args.in_no != state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1]:
#                increase_ratio = math.ceil(args.in_no / state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1])
#                state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'] = state_dict['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].repeat(1,increase_ratio,1,1)[:,:args.in_no]
#                # Figure out the position of the final layer with depth
#                final_layer = (2*args.depth)+2-1    # -1 for python indexing
#                increase_ratio = math.ceil(args.out_no / state_dict[f"UDChain.layers.{final_layer}.conv.weight"].shape[0])
#                state_dict[f"UDChain.layers.{final_layer}.conv.weight"] = state_dict[f"UDChain.layers.{final_layer}.conv.weight"].repeat(increase_ratio,1,1,1)[:args.out_no]
#                state_dict[f"UDChain.layers.{final_layer}.conv.bias"] = state_dict[f"UDChain.layers.{final_layer}.conv.bias"].repeat(increase_ratio)[:args.out_no]
#            self.model.load_state_dict(state_dict)
#        self.criterion = tools.loss.Smooth_L1_pl(reduction="mean")
#
#    def configure_optimizers(self):
#        optimizer = radam.RAdam([p for p in self.parameters() if p.requires_grad], lr=1e-6)#, weight_decay=1e-5)
#        return optimizer
#
#    def forward(self, x):
#        # Through the encoder
#        out, _ = self.model(x)
#        return out
#
#    def training_step(self, train_batch, batch_idx):
#        frame, gt_frame, vid_name, _ = train_batch
#        frame, gt_frame = frame.float(), gt_frame.float()
#        frames = frame.repeat(1,self.args.in_no,1,1)
#        out = self(frames)
#        train_loss = self.criterion(out, gt_frame)
#        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
#        return train_loss
#
#    def validation_step(self, valid_batch, batch_idx):
#        frame, gt_frame, vid_name, _ = valid_batch
#        frame, gt_frame = frame.float(), gt_frame.float()
#        frames = frame.repeat(1,self.args.in_no,1,1)
#        out = self(frames)
#        valid_loss = self.criterion(out, gt_frame)
#        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
#        return valid_loss


if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Run specific arguments")
    parser.add_argument("--task", type=str, choices=["mnist","mocap","hdmb51","pendulum-regress","roller-regress","roller-pred","segmentation","3dbounces-regress","2dbounces-regress","bounces-pred","grav-regress","grav-pred","moon-regress","blocks-regress"], help="Which task, classification or otherwise, to apply")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=1, help="minimum number of epochs to run.")
    parser.add_argument("--early_stopping", type=int, default=-1, help="number of epochs after no improvement before stopping, -1 to disable")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
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
    parser.add_argument("--disable_preload", action="store_true", help="stop the preloading of the dataset object")

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
        pl_system = FcUpDown2D2Scalars(args)

    ################################
    # Segmentation
    ################################
    elif args.task == "segmentation":
        """
        Point the dataset to the root directory
        """
        raise NotImplementedError("Not implemented segmentation")
        copy_args = copy.deepcopy(args)
        copy_args.in_no = 1
        train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', copy_args, segmentation_flag=True)
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FCUpDown2D_2_Segmentation(args)

    ################################
    # Roller regression/prediction
    ################################   
    elif args.task in ["roller-regress","roller-pred"]:
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="roller")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="roller")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)

    ################################
    # Moon regression/prediction
    ################################   
    elif args.task in ["moon-regress"]:
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="moon")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="moon")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)

    ################################
    # Block mass ratio regression
    ################################   
    elif args.task in ["blocks-regress"]:
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="blocks")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="blocks")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)

    ################################
    # Pendulum
    ################################   
    elif args.task == "pendulum-regress":
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="pendulum")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="pendulum")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)

    ################################
    # Gravity regression/prediction
    ################################   
    elif args.task in ["grav-regress","grav-pred"]:
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="grav")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="grav")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)

    ################################
    # Ball bounces regression/prediction
    ################################   
    elif args.task in ["2dbounces-regress"]:
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="2dbounces")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="2dbounces")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)

    elif args.task in ["3dbounces-regress"]:
        if args.disable_preload:
            train_dset = Simulations(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="3dbounces")
        else:
            train_dset = SimulationsPreloaded(args.dataset_path[0], 'train', 'consecutive', args, yaml_return="3dbounces")
        valid_dset = train_dset.clone('val', 'consecutive')
        test_dset = train_dset.clone('test', 'consecutive')
        pl_system = FcUpDown2D2Scalars(args)



    ################################
    # HDMB-51
    ################################
    elif args.task == "hdmb51":
        raise NotImplementedError(f"HDMB-51 dataset not been handled yet")
        #HDMB_create_labels() # TODO officially allow this function below
        train_labels, class2id, id2class = utils.load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))
        test_labels, _, _ = utils.load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))

        # Models
        model = FCUpDown2D_2_MNIST(args, load_path=args.model_path)
        model.to(args.device)

        # Training and Testing Loops
        if args.visdom:
            #args.plotter = VisdomLinePlotter(env_name=args.jobname)
            wandb.init(project="visual-modelling", entity="visual-modelling", name=args.jobname)
            wandb.config.update(args)
        # Training (includes validation after each epoch and early stopping)
        if args.epoch > 0:        
            best_acc, return_string = train_HDMB_51(model, args, bsz=args.bsz, epochs=args.epoch)
        # Only Testing
        if args.epoch == 0:
            best_acc, _ = test_HDMB_51(model, args, bsz=args.val_bsz)
            return_string = f"Validation Only: Accuracy: {best_acc:.2f}%"

    ################################
    # MOCAP
    ################################
    elif args.task == "mocap":
        raise NotImplementedError(f"MOCAP split currently gathered has not proven useful enough to use")
        #MOCAP_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/mocap/labels.txt")
        #pandas_mocap = pd.read_csv(MOCAP_labels_path, sep="\t")
        #label_dict = {}
        #for key, row in pandas_mocap.iterrows():
        #    subj, descr = row["Nested_ID"], row["class/description"]
        #    if subj[0]=="#":
        #        descr = descr.replace("(","").replace(")","")
        #        sub_dict = {"class/description": descr}
        #        subj_counter = f"{int(subj[1:]):02}"
        #        print(int(subj[1:]))
        #        label_dict[subj_counter] = sub_dict
        #    else:
        #        label_dict[subj_counter][f"{int(subj):02}"] = descr
        #print(label_dict)

    train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=args.shuffle)
    valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=args.val_bsz, num_workers=args.num_workers, shuffle=False)

    # Checkpointing and running
    if args.task in ["mnist","mocap","hdmb51","grav-pred","bounces-pred","roller-pred"]:   # Accuracy tasks
        max_or_min = "max"
        monitoring = "valid_acc"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitoring,
            dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
            filename=f"{args.jobname}"+'-{epoch:02d}-{valid_acc:.2f}',
            save_top_k=1,
            mode=max_or_min,
        )
    elif args.task in ["segmentation","pendulum-regress","3dbounces-regress","2dbounces-regress","grav-regress","roller-regress","moon-regress","blocks-regress"]:
        max_or_min = "min"
        monitoring = "valid_loss"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitoring,
            dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
            filename=f"{args.jobname}"+'-{epoch:02d}-{valid_loss:.3f}',
            save_top_k=1,
            mode=max_or_min,
        )
    else:
        raise NotImplementedError(f"Task: {args.task} is not handled")

    if args.early_stopping >= 0:
        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor=monitoring, patience=args.early_stopping, mode=max_or_min)
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        callbacks = [checkpoint_callback]

    pl_system.testing = False
    trainer = pl.Trainer(callbacks=callbacks, logger=wandb_logger, gpus=gpus, max_epochs=args.epoch, min_epochs=args.min_epochs)
    trainer.fit(pl_system, train_loader, valid_loader)
    pl_system.testing = True
    trainer.test(test_dataloaders=test_loader, ckpt_path='best')



################################
# Utility Functions
################################
#def HDMB_create_labels():
#    HDMB51_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/testTrainMulti_7030_splits")
#    HDMB51_labels = os.listdir(HDMB51_labels_path)
#    train = {}
#    test = {}
#    other = {}
#    for txtfile in HDMB51_labels:
#        fname = txtfile.split(".")[0]
#        ctgry, _ = fname.split("_test_split")
#        #ctrgy, splt = fname.split("_test_split")
#        assert len(fname.split("_test_split")) == 2, f"{txtfile} breaks this formatting"
#        #splt = splt.split(".")[0]
#        
#        labs = open(f"{HDMB51_labels_path}/{txtfile}", "r").readlines()
#        for ele in labs:
#            vid, train_or_test =  ele.split()
#            vid = vid.split(".")[0]                
#            if train_or_test == "0":
#                other[vid] = ctgry
#            elif train_or_test == "1":
#                train[vid] = ctgry
#            elif train_or_test == "2":
#                test[vid] = ctgry
#            else:
#                print(f"Should not have happend: Test-or-test ID:{train_or_test}")
#    sorted_lab_type = sorted(set(train.values()))
#    class2id = { lab:idx for idx, lab in enumerate(sorted_lab_type)}
#    id2class = { idx:lab for idx, lab in enumerate(sorted_lab_type)}
#    train   = ({ vid:class2id[lab] for vid, lab in train.items()}, class2id, id2class)
#    test    = ({ vid:class2id[lab] for vid, lab in test.items()}, class2id, id2class)
#    other   = ({ vid:class2id[lab] for vid, lab in other.items()}, class2id, id2class)
#    utils.save_pickle(train, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))
#    utils.save_pickle(test , os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/test_labels.pickle"))
#    utils.save_pickle(other, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/other_labels.pickle"))
