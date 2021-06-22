__author__ = "Jumperkables"

import os
import sys
import argparse
import shutil
import copy
import wandb
import piq
import imageio
import torchmetrics.functional
import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from cv2 import putText, FONT_HERSHEY_SIMPLEX
from tqdm import tqdm

# local
import tools.loss
import tools.radam as radam
import tools.utils as utils
from dataset import MMNIST, Simulations
from tools.utils import model_fwd
from tools.ball_distance_metric import calculate_metric
from models.UpDown2D import FCUpDown2D
from models.transformer import ImageTransformer
from models.deans_transformer import VMDecoder as DeansTransformer



################################################################################
#### UTILITY FUNCTIONS
################################################################################
def plot_self_out(pl_system):
    args = pl_system.args
    # Remove all previous gifs
    [ os.remove(os.path.join(args.results_dir, file)) for file in os.listdir(args.results_dir) if file.endswith('.gif') ]
    self_out_loader = pl_system.self_out_loader
    wandb_frames = []
    wandb_metric_n_names = []
    metrics = {
        'psnr':torchmetrics.functional.psnr,
        'sl1':F.smooth_l1_loss,
        #'LPIPS':piq.LPIPS(), # TODO Re-add this, it is very slow, maybe not worth it
        'ssim':torchmetrics.functional.ssim,
        'MS_SSIM':piq.multi_scale_ssim,
        'ball_distance':calculate_metric,
        #'FID':piq.FID(),
        #'FVD':tools.loss.FVD()
    }
    with tqdm(total=args.n_gifs) as pbar:
        for ngif in range(args.n_gifs):
            pbar.update(1)
            pbar.set_description(f"Self output: {ngif+1}/{args.n_gifs}")
            start_frames, gt_frames, vid_name, _ = next(iter(self_out_loader))
            start_frames = start_frames.float().to(pl_system.device)
            out = pl_system(start_frames)
            gif_frames = []
            if args.self_output_n == -1:
                self_output_n = gt_frames.shape[1]
            else:
                self_output_n = args.self_output_n
            for itr in range(0, self_output_n, args.out_no):
                start_frames = torch.cat([ start_frames[:,args.out_no:args.in_no] , out ], 1)
                out = pl_system(start_frames)
                for n in range(args.out_no):
                    gif_frames.append(out[0][n].cpu().detach().byte())
                # Add the ground truth frame side by side to generated frame
            gif_frames = gif_frames[:gt_frames.shape[1]]
            gif_metrics = get_gif_metrics(gif_frames, gt_frames, metrics)
            colour_gradients = [255,240,225,210,195,180,165,150,135,120,120,135,150,165,180,195,210,225,240,255]   # Make sure that white/grey backgrounds dont hinder the frame count
            # Number the frames to see which frame of the gif in output plut
            gt_frames = [torch.from_numpy(putText(np.array(frame), f"{f_idx}", (0,frame.shape[1]), FONT_HERSHEY_SIMPLEX, fontScale = 0.55, color = (colour_gradients[f_idx%len(colour_gradients)]))) for f_idx, frame in enumerate(gt_frames[0])]

            # Ball distance plot
            img_h = start_frames.shape[2]
            # TODO Be sure that this dimension is height, not width
            bdm = np.array(gif_metrics["ball_distance"]).astype(np.double)
            bdm_mask = np.isfinite(bdm)
            bdm = bdm[bdm_mask]
            plt.plot(bdm)
            canvas = plt.gcf()
            dpi = plt.gcf().get_dpi()
            canvas.set_size_inches(2*img_h/dpi, 2*img_h/dpi)
            canvas.suptitle(f"Ball Distance", fontsize=6)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            canvas.tight_layout()
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            ball_distance_image = data.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.clf()
            ball_distance_image = torch.from_numpy(np.copy(ball_distance_image)).permute(2,0,1)#.unsqueeze(0)
            ball_distance_image = ball_distance_image.float().mean(0).byte()

            # SSIM plot
            plt.plot(gif_metrics['ssim'])
            canvas = plt.gcf()
            dpi = plt.gcf().get_dpi()
            canvas.set_size_inches(2*img_h/dpi, 2*img_h/dpi)
            canvas.suptitle(f"{vid_name[0]}:SSIM", fontsize=6)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            canvas.tight_layout()
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            ssim_image = data.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.clf()
            ssim_image = torch.from_numpy(np.copy(ssim_image)).permute(2,0,1)#.unsqueeze(0)
            ssim_image = ssim_image.float().mean(0).byte()

            # SL1 plot
            plt.plot(gif_metrics['sl1'])
            canvas = plt.gcf()
            dpi = plt.gcf().get_dpi()
            canvas.set_size_inches(2*img_h/dpi, 2*img_h/dpi)
            canvas.suptitle("SL1", fontsize=6)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            canvas.tight_layout()
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            sl1_image = data.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.clf()
            sl1_image = torch.from_numpy(np.copy(sl1_image)).permute(2,0,1)#.unsqueeze(0)
            sl1_image = sl1_image.float().mean(0).byte()

            # Gif
            gif_frames = [ torch.cat( [torch.cat( [torch.stack(gif_frames)[n_frm], gt_frames[n_frm]], dim=0), ball_distance_image, ssim_image, sl1_image], dim=1)for n_frm in range(len(gif_frames)) ]
            gif_save_path = os.path.join(args.results_dir, f"{ngif}-{vid_name[0]}.gif") 
            # TODO gifs from different datasets with the same name will overwrite eachother. this is niche and not worth the time right now
            imageio.mimsave(gif_save_path, gif_frames)
            wandb_frames.append(wandb.Video(gif_save_path))
            gif_metrics['name'] = vid_name[0]
            # TODO Deprecated?
            #wandb_metric_n_names.append(gif_metrics)
        pbar.close()
    wandb.log({"self_output_gifs": wandb_frames}, commit=False)
    wandb.log({"metrics":wandb_metric_n_names}, commit=True)


def get_gif_metrics(gif_frames, gt_frames, metrics):
    gif_frames, gt_frames = torch.stack(gif_frames).unsqueeze(1), gt_frames[0].unsqueeze(1)
    # TODO Can use this if i want the old metrics
    #metric_vals = {
    #    'psnr':float(metrics['psnr'](gif_frames, gt_frames)),
    #    #'LPIPS':float(metrics['LPIPS'](gif_frames.float(), gt_frames.float())), #TODO re-include this later if we want
    #    'ssim':float(metrics['ssim'](gif_frames.float(), gt_frames.float())),
    #    'sl1':float(metrics['sl1'](gif_frames.float(), gt_frames.float())),
    #    'MS_SSIM':-1. if (gif_frames.shape[2]<=161 and gif_frames.shape[3]<=161) else float(metrics['MS_SSIM'](gif_frames, gt_frames, data_range=255.)),
    #    #'FID':float(metrics['FID'](metrics['FID']._compute_feats(DataLoader(FID_dset(gif_frames.float()))), metrics['FID']._compute_feats(DataLoader(FID_dset(gt_frames.float()))))),
    #    #'FVD':metrics['FVD'](gif_frames[:utils.round_down(gif_frames.shape[0],16)], gt_frames[:utils.round_down(gt_frames.shape[0],16)] )
    #    #'FVD':metrics['FVD'](gif_frames, gt_frames)
    #}
    #running_psnr = []
    running_ball_distance = []
    running_ssim = []
    running_sl1 = []
    for frame_idx in range(gif_frames.shape[0]):
        #running_psnr.append( float(metrics['psnr']( gif_frames[frame_idx].float(), gt_frames[frame_idx].float())) )
        running_ball_distance.append( metrics['ball_distance']( gif_frames[frame_idx], gt_frames[frame_idx]) )
        running_ssim.append( float(metrics['ssim']( gif_frames[frame_idx].unsqueeze(0).float(), gt_frames[frame_idx].unsqueeze(0).float())) )
        running_sl1.append( float(metrics['sl1']( gif_frames[frame_idx].float(), gt_frames[frame_idx].float())) )
    metric_vals = {
        #'psnr':running_psnr,
        'ball_distance':running_ball_distance,
        'ssim':running_ssim,
        'sl1':running_sl1
    }
    #raise NotImplementedError("Make sure that these diverging metrics are calculated correctly")
    return metric_vals


#### Wrapper classes for FID loss
class FID_dset(torch.utils.data.Dataset):
    def __init__(self, frames):
        self.frames = frames
    def __len__(self):
        return(self.frames.shape[0])
    def __getitem__(self, idx):
        return self.frames(idx)





################################################################################
################################################################################
#### PYTORCH LIGHTNING MODULES
#### ADD YOURS HERE
################################################################################
################################################################################
class ModellingSystem(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, self_out_loader):
        """
        self_out_loader:    A pytorch dataloader specialised for the self-output generation
        """
        super().__init__()
        self.args = args
        self.self_out_loader = self_out_loader

        # Model selection
        if args.model == "UpDown2D":
            self.model = FCUpDown2D(args)
        elif args.model == "image_transformer":
            self.model = ImageTransformer(args)
        elif args.model == "deans_transformer":
            self.model = DeansTransformer(in_dim=args.d_model, layers=args.n_layers, heads=args.nhead)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # args.reduction == none requires manual optimisation flag set
        if args.reduction == "none":
            self.automatic_optimization = False 

        # Validation metrics
        self.valid_PSNR = torchmetrics.functional.psnr
        self.valid_SSIM = torchmetrics.functional.ssim
        self.valid_sl1 = nn.SmoothL1Loss(reduction=args.reduction)#.to(self.device)
        #self.valid_focal = tools.loss.FocalLoss().to(self.device)
        # TODO Remove this workaround when 'on_best_epoch' is implemented in lightning
        self.best_loss = float('inf')

        # Criterion and plotting loss
        if args.loss == "mse":
            self.valid_loss = torchmetrics.functional.mean_squared_error
            self.train_loss = torchmetrics.functional.mean_squared_error
            self.criterion =  torchmetrics.functional.mean_squared_error
        elif args.loss == "focal":
            raise NotImplementedError("Need to implement this for focal loss")
            self.valid_loss = None
            self.train_loss = None
        elif args.loss == "sl1":
            self.valid_loss = nn.SmoothL1Loss(reduction=args.reduction)#.to(self.device)
            self.train_loss = nn.SmoothL1Loss(reduction=args.reduction)#.to(self.device)
            self.criterion = nn.SmoothL1Loss(reduction=args.reduction)#.to(self.device)
        elif args.loss == "ssim":
            self.valid_loss = torchmetrics.functional.ssim
            self.train_loss = torchmetrics.functional.ssim
            self.criterion = torchmetrics.functional.ssim
        else:
            raise ValueError(f"Unknown loss: {args.loss}")
        
    def forward(self, x):
        out, _ = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = radam.RAdam([p for p in self.parameters() if p.requires_grad], lr=self.args.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        frames, gt_frames, vid_names, _ = train_batch
        frames, gt_frames = frames.float(), gt_frames.float()
        out = self(frames)
        train_loss = self.criterion(out, gt_frames)
        if self.args.reduction == 'none':
            grad = torch.ones(train_loss.shape, requires_grad=True).to(self.device)
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(train_loss, gradient=grad)
            opt.step()
        if self.args.loss == "ssim":
            train_loss = 1-((1+train_loss)/2)   # SSIM Range = (-1 -> 1) SSIM should be maximised => restructure as minimisation
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, valid_batch, batch_idx):
        frames, gt_frames, vid_names, _ = valid_batch
        frames, gt_frames = frames.float(), gt_frames.float()
        out = self(frames)
        valid_loss = self.criterion(out, gt_frames)
        if self.args.reduction == 'none':
            valid_loss = valid_loss.mean(dim=(0,1,2,3))
        if self.args.loss == "ssim":
            valid_loss = 1-((1+valid_loss)/2)   # SSIM Range = (-1 -> 1) SSIM should be maximised => restructure as minimisation
        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid_PSNR", self.valid_PSNR(out, gt_frames), on_step=False, on_epoch=True)
        self.log("valid_SSIM", self.valid_SSIM(out, gt_frames), on_step=False, on_epoch=True)
        self.log("valid_sl1", self.valid_sl1(out, gt_frames), on_step=False, on_epoch=True)
        #self.log("valid_focal", self.valid_focal(out, gt_frames), on_step=False, on_epoch=True)
        return valid_loss

    def validation_epoch_end(self, validation_step_outputs):
        # TODO update this with 'on_best_epoch' functionality when it is supported
        # TODO compute_on_step functionality isn't working. update when it is
        valid_loss = float(sum(validation_step_outputs)/len(validation_step_outputs))
        if (valid_loss < self.best_loss):
            self.log("best_epoch", wandb.Html(str(self.current_epoch)))
            if not self.trainer.running_sanity_check:  # Dont adjust loss for initial sanity check
                self.best_loss = float(valid_loss)
            plot_self_out(self)  # Plot gifs and metrics


if __name__ == "__main__":
    #torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Run specific arguments")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=2, help="number of epochs after no improvement before stopping")
    """
    Guide to split_condition:
        'tv_ratio:4-1' : Simply split all videos into train:validation ratio of 4:1

    """
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--num_workers", type=int, default=0, help="Pytorch dataloader workers")
    parser.add_argument("--wandb", action="store_true", help="Use wandb plotter")
    parser.add_argument("--wandb_entity", type=str, default="visual-modelling", help="wandb entity to save project and run in")
    parser.add_argument("--n_gifs", type=int, default=10, help="Number of output gifs to visualise")
    parser.add_argument("--self_output_n", type=int, default=-1, help="Number of frames to run selfoutput plotting to. -1 = all available frames")
    parser.add_argument("--jobname", type=str, required=True, help="Jobname for wandb and saving things")

    parser.add_argument_group("Dataset specific arguments")
    ############# To combine multiple datasets together, align the dataset and dataset path arguments
    parser.add_argument("--dataset", type=str, nargs="+", required=True, choices=["mmnist", "simulations", "mocap", "hdmb51"], help="Type of dataset")
    parser.add_argument("--dataset_path", type=str, nargs="+", default=os.path.expanduser("~/"), help="Dataset paths")
    #############
    parser.add_argument("--split_condition", type=str, default="tv_ratio:4-1", help="Custom string deciding how to split datasets into train/test. Affiliated with a custom function in dataset")
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")
    
    parser.add_argument_group("Shared Model argmuents")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "image_transformer", "deans_transformer"], help="Type of model to run")

    parser.add_argument_group("2D and 3D CNN specific arguments")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
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

    parser.add_argument_group("Other things")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "sl1", "focal", "ssim"], help="Loss function for the network")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum", "none"], help="type of reduction to apply on loss")
    parser.add_argument("--lr", type=float, default=3e-4, help="Set the learning rate for the RAdam Optimiser")

    ####
    # Sorting arguements
    args = parser.parse_args()
    print(args)

    ######## ERROR CONDITIONS To make sure erroneous runs aren't accidentally executed
    # This code allows multiple datasets to be combined, assert this has happened correctly
    assert len(args.dataset) == len(args.dataset_path), f"Number of specified dataset paths and dataset types should be equal"

    # Mean squared error is naturally only ran with reduction of mean
    if args.loss == "mse":
        assert args.reduction == "mean", f"MSE loss only works with reduction set to mean"
    if (not args.shuffle) and (len(args.dataset_path)>1):
        raise NotImplementedError("Shuffle because multiple self_out data examples from each dataset need to be represented")
    
    # SSIM functional needs to be used. reduction cannot be specified
    if args.loss == "ssim":
        assert args.reduction == "mean", f"SSIM functional needs to be used. cant be bothered to rewrite to allow this for now as its irrelevant. instead default to mean reduction"
    ########

    #### Make sure the dataset object is configured properly
    dataset_switch = {
        "simulations": Simulations,
        "mmnist" : Simulations,  # This may change one day, but this works just fine
        "mocap" : Simulations,
        "hdmb51" : Simulations,
    }
    dataset_list = args.dataset_path
    train_list = []
    valid_list = []
    self_out_list = []
    print(f"\nProcessing {args.dataset_path} datasets...")
    for i in tqdm(range(len(args.dataset))):
        train_dset = dataset_switch[args.dataset[i]](args.dataset_path[i], args)
        valid_dset = copy.deepcopy(train_dset)
        valid_dset.set_mode("valid")
        valid_list.append(valid_dset)
        self_out_dset = copy.deepcopy(train_dset)
        self_out_dset.set_mode("self_out")
        self_out_dset = torch.utils.data.Subset(self_out_dset, [i for i in range(args.n_gifs)]) # Only have the number of gifs required
        self_out_list.append(self_out_dset)
        train_dset.set_mode("train")
        train_list.append(train_dset)
    if len(args.dataset) > 1:
        train_dset = torch.utils.data.ConcatDataset(train_list)
        valid_dset = torch.utils.data.ConcatDataset(valid_list)
        self_out_dset = torch.utils.data.ConcatDataset(self_out_list)
    else:
        train_dset = train_list[0]
        valid_dset = valid_list[0]
        self_out_dset = self_out_list[0]
    pin_memory = args.device >= 0 and args.num_workers >= 1
    train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=args.shuffle, pin_memory=pin_memory)#, drop_last=True)
    valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, shuffle=args.shuffle, pin_memory=pin_memory)#, drop_last=True)
    torch.manual_seed(2667)
    self_out_loader = DataLoader(self_out_dset, batch_size=1, num_workers=args.num_workers, shuffle=args.shuffle, drop_last=True, pin_memory=pin_memory)

    #### Logging and Saving: If we're saving this run, prepare the neccesary directory for saving things
    wandb.init(entity=args.wandb_entity, project="visual-modelling", name=args.jobname)
    wandb_logger = pl.loggers.WandbLogger(offline=not args.wandb)#, resume="allow")
    wandb_logger.log_hyperparams(args)
    repo_rootdir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_dir = os.path.join(repo_rootdir, ".results", args.jobname )
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    args.results_dir = results_dir

    # Model info
    if args.model == "UpDown3D":
        raise NotImplementedError("Move 3D CNN to Pytorch lightning")
        #pl_system = FCUp_Down3D(args)
    elif args.model == "UpDown2D":
        pl_system = ModellingSystem(args, self_out_loader)
    elif args.model == "image_transformer":
        pl_system = ModellingSystem(args, self_out_loader)
    elif args.model == "deans_transformer":
        pl_system = ModellingSystem(args, self_out_loader)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # GPU
    if args.device == -1:
        gpus = None 
    else: 
        gpus = [args.device]  # TODO Implement multi GPU support

    # Checkpointing and running
    if args.loss in ["mse", "sl1", "focal"]:  
        max_or_min = "min"  # Minimise these validation losses
        monitoring = "valid_loss"   # And monitor validation loss
    elif args.loss == "ssim":
        max_or_min = "min"
        monitoring = "valid_loss"
    else:
        raise NotImplementedError(f"Loss: {args.loss} not implemented.")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitoring,
        dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
        filename=f"{args.jobname}"+'-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=1,
        mode=max_or_min,
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=wandb_logger, gpus=gpus, max_epochs=args.epoch)
    trainer.fit(pl_system, train_loader, valid_loader)
