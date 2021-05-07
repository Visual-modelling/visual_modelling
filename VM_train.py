__author__ = "Jumperkables"

import os, sys, argparse, shutil, copy
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from cv2 import putText, FONT_HERSHEY_SIMPLEX

from dataset import MMNIST, Simulations
from models.UpDown2D import FCUp_Down2D, FCUp_Down2D_2_MNIST, FCUp_Down2D_2_Segmentation, FCUp_Down2D_2_FCPred
from models.UpDown3D import FCUp_Down3D
from models.transformer import VMTransformer, VMTransformer2 
#from models.OLD_transformer import VMTransformer, VMTransformer2 
import tools.utils as utils
from tools.utils import model_fwd
import tools.radam as radam
import tools.loss
import torch.nn.functional as F
import tqdm
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics

import wandb
import piq

import imageio


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
        #'FID':piq.FID(),
        #'FVD':tools.loss.FVD()
    }
    with tqdm(total=args.n_gifs) as pbar:
        for ngif in range(args.n_gifs):
            pbar.update(1)
            pbar.set_description(f"Self output: {ngif+1}/{args.n_gifs}")
            start_frames, gt_frames, vid_name = next(iter(self_out_loader))
            start_frames = start_frames.float().to(args.device)
            out = pl_system(start_frames)
            gif_frames = []
            if args.self_output_n == -1:
                self_output_n = gt_frames.shape[1]
            else:
                self_output_n = args.self_output_n
            for itr in range(self_output_n):
                start_frames = torch.cat([ start_frames[:,args.out_no:args.in_no] , out ], 1)
                out = pl_system(start_frames)
                for n in range(args.out_no):
                    gif_frames.append(out[0][n].cpu().detach().byte())
                # Add the ground truth frame side by side to generated frame
            gif_metrics = get_gif_metrics(gif_frames, gt_frames, metrics)
            colour_gradients = [255,240,225,210,195,180,165,150,135,120,120,135,150,165,180,195,210,225,240,255]   # Make sure that white/grey backgrounds dont hinder the frame count
            # Number the frames to see which frame of the gif in output plut
            gt_frames = [torch.from_numpy(putText(np.array(frame), f"{f_idx}", (0,frame.shape[1]), FONT_HERSHEY_SIMPLEX, fontScale = 0.55, color = (colour_gradients[f_idx%len(colour_gradients)]))) for f_idx, frame in enumerate(gt_frames[0])]
            # PSNR plot
            img_h = start_frames.shape[2]
            # TODO Be sure that this dimension is height, not width
            plt.plot(gif_metrics["psnr"])
            canvas = plt.gcf()
            dpi = plt.gcf().get_dpi()
            canvas.set_size_inches(2*img_h/dpi, 2*img_h/dpi)
            canvas.suptitle(f"PSNR", fontsize=6)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            canvas.tight_layout()
            canvas = plt.gca().figure.canvas
            canvas.draw()
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            psnr_image = data.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.clf()
            psnr_image = torch.from_numpy(np.copy(psnr_image)).permute(2,0,1)#.unsqueeze(0)
            psnr_image = psnr_image.float().mean(0).byte()

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
            gif_frames = [ torch.cat( [torch.cat( [torch.stack(gif_frames)[n_frm], gt_frames[n_frm]], dim=0), psnr_image, ssim_image, sl1_image], dim=1)for n_frm in range(len(gif_frames)) ]
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
    #import ipdb; ipdb.set_trace()
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
    running_psnr = []
    running_ssim = []
    running_sl1 = []
    for frame_idx in range(gif_frames.shape[0]):
        running_psnr.append( float(metrics['psnr']( gif_frames[frame_idx].float(), gt_frames[frame_idx].float())) )
        running_ssim.append( float(metrics['ssim']( gif_frames[frame_idx].unsqueeze(0).float(), gt_frames[frame_idx].unsqueeze(0).float())) )
        running_sl1.append( float(metrics['sl1']( gif_frames[frame_idx].float(), gt_frames[frame_idx].float())) )
    metric_vals = {
        'psnr':running_psnr,
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
class Bouncing_CNN(pl.LightningModule):
    """
    self_out_loader:    A pytorch dataloader specialised for the self-output generation
    """
    def __init__(self, args, self_out_loader):
        super().__init__()
        self.args = args
        self.model = FCUp_Down2D(args)
        self.self_out_loader = self_out_loader

        # Validation metrics
        self.valid_PSNR = torchmetrics.functional.psnr
        self.valid_SSIM = torchmetrics.functional.ssim #torchmetrics.SSIM() # TODO reintroduce when memory leak is solved https://github.com/PyTorchLightning/pytorch-lightning/issues/5733
        self.valid_LPIPS = tools.loss.LPIPS_Metric_pl()
        self.valid_MS_SSIM = tools.loss.MS_SSIM_Metric_pl()
        # TODO Remove this workaround when 'on_best_epoch' is implemented in lightning
        self.best_loss = float('inf')

        # Training metrics
        self.train_PSNR = torchmetrics.functional.psnr
        self.train_SSIM = torchmetrics.functional.ssim
        self.train_LPIPS = tools.loss.LPIPS_Metric_pl()
        self.train_MS_SSIM = tools.loss.MS_SSIM_Metric_pl()

        # Criterion and plotting loss
        ## TODO Get pytorch lightning compatible metrics when they become of available for all below
        if args.loss == "mse":  # Only compatible with mean reduction https://torchmetrics.readthedocs.io/en/latest/references/modules.html#meansquarederror
            self.valid_loss = torchmetrics.functional.mean_squared_error
            self.train_loss = torchmetrics.functional.mean_squared_error
            self.criterion =  torchmetrics.functional.mean_squared_error
        elif args.loss == "focal":
            raise NotImplementedError("Need to implement this for focal loss")
            self.valid_loss = None
            self.train_loss = None
            #self.criterion = tools.loss.FocalLoss(reduce=args.reduce, reduction=args.reduction).to(args.device) # TODO Get torchmetrics focal loss when it exists
        elif args.loss == "sl1":
            self.valid_loss = tools.loss.Smooth_L1_pl(reduction=args.reduction)
            self.train_loss = tools.loss.Smooth_L1_pl(reduction=args.reduction)
            self.criterion = tools.loss.Smooth_L1_pl(reduction=args.reduction)
        elif args.loss == "ssim":
            self.valid_loss = torchmetrics.functional.ssim#SSIM(reduction="elementwise_mean" if args.reduction == "mean" else args.reduction) # TODO when fixed, use torchmetrics
            self.train_loss = torchmetrics.functional.ssim#(reduction="elementwise_mean" if args.reduction == "mean" else args.reduction)
            self.criterion = torchmetrics.functional.ssim#SSIM(reduction="elementwise_mean" if args.reduction == "mean" else args.reduction)
            #tools.loss.ssim(data_range=255, size_average=True, channel=1).to(args.device)
        else:
            raise Exception("Loss not implemented")
        
    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = radam.RAdam([p for p in self.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        frames, gt_frames, vid_names = train_batch
        frames, gt_frames = frames.float(), gt_frames.float()
        out = self(frames)
        train_loss = self.criterion(out, gt_frames)
        if self.args.loss == "ssim":
            train_loss = 1-((1+train_loss)/2)   # SSIM Range = (-1 -> 1) SSIM should be maximised => restructure as minimisation
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, valid_batch, batch_idx):
        frames, gt_frames, vid_names = valid_batch
        frames, gt_frames = frames.float(), gt_frames.float()
        out = self(frames)
        valid_loss = self.criterion(out, gt_frames)
        if self.args.loss == "ssim":
            valid_loss = 1-((1+valid_loss)/2)   # SSIM Range = (-1 -> 1) SSIM should be maximised => restructure as minimisation
        self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss

    def validation_epoch_end(self, validation_step_outputs):
        # TODO update this with 'on_best_epoch' functionality when it is supported
        # TODO compute_on_step functionality isn't working. update when it is
        valid_loss = float(sum(validation_step_outputs)/len(validation_step_outputs))
        if (valid_loss < self.best_loss):
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
    parser.add_argument("--segmentation", action="store_true", help="Create a dataset for image segmentation. Segmentation masks for images in video clips should be named the same as the original image and stored in a subdirectory of the clip 'mask'")
    parser.add_argument("--grav_pred", action="store_true", help="Gravity prediction task")
    parser.add_argument("--bounces_pred", action="store_true", help="Bounce counting task")

    parser.add_argument_group("2D and 3D CNN specific arguments")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "transformer"], help="Type of model to run")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--krnl_size", type=int, default=3, help="Height and width kernel size")
    parser.add_argument("--krnl_size_t", type=int, default=3, help="Temporal kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Height and width Padding")
    parser.add_argument("--padding_t", type=int, default=1, help="Temporal Padding")
    parser.add_argument("--depth", type=int, default=2, help="depth of the updown")
    parser.add_argument("--channel_factor", type=int, default=64, help="channel scale factor for up down network")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "sl1", "focal", "ssim"], help="Loss function for the network")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")

    ####
    # Sorting arguements
    args = parser.parse_args()
    print(args)

    ######## ERROR CONDITIONS To make sure erroneous runs aren't accidentally executed
    ## Only one task should be running
    task_count = (args.segmentation)+(args.grav_pred)+(args.bounces_pred)
    assert task_count<=1, f"You can only run one of gravity prediction, segmentation, or bounce prediction"

    # Segmentation as a task is 1 image in and 1 out
    if args.segmentation:
        assert args.in_no == args.out_no == 1, f"Image segmentation is defined on 1 input and 1 output image. Not {args.in_no} and {args.out_no}"

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
    train_loader = DataLoader(train_dset, batch_size=args.bsz, num_workers=args.num_workers, shuffle=args.shuffle, drop_last=True)
    valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz, num_workers=args.num_workers, shuffle=args.shuffle, drop_last=True)
    self_out_loader = DataLoader(self_out_dset, batch_size=1, num_workers=args.num_workers, shuffle=args.shuffle, drop_last=True)

    #### Logging and Saving: If we're saving this run, prepare the neccesary directory for saving things
    wandb_logger = pl.loggers.WandbLogger(project="visual-modelling", name=args.jobname, offline=not args.wandb)#, resume="allow")
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
        pl_system = Bouncing_CNN(args, self_out_loader)
        #if args.segmentation:
        #    model = FCUp_Down2D_2_Segmentation(args, args.model_path, load_mode="pad")
        #elif args.grav_pred:
        #    model = FCUp_Down2D_2_FCPred(args, args.model_path, mode="grav_pred")
        #elif args.bounces_pred:
        #    model = FCUp_Down2D_2_FCPred(args, args.model_path, mode="bounces_pred")
        #else:
        #    model = FCUp_Down2D(args)#.depth, args.in_no, args.out_no, args)
    elif args.model == "transformer":
        breakpoint()
        print("Your transformer model here")
        pl_system = None 
        import sys; sys.exit()
    else:
        raise Exception("Model: %s not implemented" % (args.model))

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
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=wandb_logger, gpus=gpus)
    trainer.fit(pl_system, train_loader, valid_loader)







    ######
    ## IGNORE FROM HERE (WILL MAKE THIS MORE ELEGANT)
    ######
    ## TODO make this overwrite of criterion more elegant
    #if args.grav_pred:
    #    criterion = torch.nn.MSELoss().to(args.device)
    #if args.bounces_pred:
    #    criterion = torch.nn.MSELoss().to(args.device)

    #    #wandb.config.update(args, allow_val_change=True)

    ## Training loop
    #early_stop_count = 0
    #early_stop_flag = False
    #best_loss = 10**20    # Mean average precision
    #if args.epoch == 0:
    #    validate(args, dset, model, criterion)
    #else:
    #    print("Training Start")
    #for epoch in range(args.epoch):
    #    if not early_stop_flag:
    #        if args.bounces_pred or args.grav_pred:
    #            train_loss, valid_loss, train_acc, valid_acc = train(args, dset, model, optimizer, criterion, epoch, best_loss)
    #        else:
    #            train_loss, valid_loss = train(args, dset, model, optimizer, criterion, epoch, best_loss)
    #        if valid_loss > best_loss:  # No improvement
    #            early_stop_count += 1
    #            if early_stop_count >= args.early_stopping:
    #                early_stop_flag = True
    #        else:                       # New Best Epoch
    #            early_stop_count  = 0
    #            best_loss = valid_loss
    #            if args.visdom:
    #                pass
    #                #args.plotter.text_plot(args.jobname+" val", "Best %s val %.4f Iteration:%d" % (args.loss, best_loss, epoch))
    #            if args.save:
    #                torch.save(model.state_dict(), args.checkpoint_path)
    #                if task_count == 0: # If this isnt segmentation/gravity prediction etc..
    #                    metrics = self_output(args, model, dset)
    #                    psnr_metric = utils.avg_list([ empl["PSNR"] for empl in metrics ])
    #                    ssim_metric = utils.avg_list([ empl["SSIM"] for empl in metrics ])
    #                    msssim_metric = utils.avg_list([ empl["MS_SSIM"] for empl in metrics ])
    #                    lpips_metric = utils.avg_list([ empl["LPIPS"] for empl in metrics ])
    #                    # Printing and logging either way
    #                    print(f"Epoch {epoch}/{args.epoch}", f"Train Loss {train_loss:.3f} |", f"Val Loss {valid_loss:.3f} |", f"Early Stop Flag {early_stop_count}/{args.early_stopping}", f"PSNR {psnr_metric:.3f}", f"SSIM {ssim_metric:.3f}", f"MS-SSIM {msssim_metric:.3f}", f"LPIPS {lpips_metric:.3f}\n")
    #                model.train()
    #            else:
    #                # Printing and logging either way
    #                print(f"Epoch {epoch}/{args.epoch}", f"Train Loss {train_loss:.3f} |", f"Val Loss {valid_loss:.3f} |", f"Early Stop Flag {early_stop_count}/{args.early_stopping}\n")
    #        if args.visdom:
    #            if args.save:
    #                if task_count == 0: # If this isnt segmentation/gravity prediction etc..
    #                    wandb.log({'PSNR' : psnr_metric}, commit=False)
    #                    wandb.log({'SSIM' : ssim_metric}, commit=False)
    #                    wandb.log({'MS-SSIM' : msssim_metric}, commit=False)
    #                    wandb.log({'LPIPS' : lpips_metric}, commit=False)
    #            wandb.log({'val_loss' : best_loss})
    #            wandb.log({'train_loss':train_loss})
    #            if args.bounces_pred or args.grav_pred:
    #                wandb.log({'train_acc' : train_acc})
    #                wandb.log({'valid_acc' : valid_acc})
    #            #args.plotter.text_plot(args.jobname+" epoch", f"Train Loss {train_loss:.3f} | Val Loss {valid_loss:.3f} | Early Stop Count {early_stop_count}")
    #            #args.plotter.plot(args.loss, "val", "val "+args.jobname, epoch, valid_loss)
    #            #args.plotter.plot(args.loss, "train", "train "+args.jobname, epoch, train_loss)

    #    else:
    #        print_string = "Early stop on epoch %d/%d. Best %s %.3f at epoch %d" % (epoch+1, args.epoch, args.loss, best_loss, epoch+1-early_stop_count)
    #        print(print_string)
    #        #args.plotter.text_plot(args.jobname+" epoch", print_string)
    #        sys.exit()

#def train(args, dset, model, optimizer, criterion, epoch, previous_best_loss):
#    # Prepare validation loader
#    #(dset, model, mode, args)
#    set_modes(dset, model, "train", args)
#    train_loader = DataLoader(dset, batch_size=args.bsz, shuffle=args.shuffle)#, drop_last=True)
#    train_loss = []
#    train_acc = []
#    with tqdm(total=len(train_loader)) as pbar:
#        for batch_idx, batch in enumerate(train_loader):
#            pbar.update(1)
#            pbar.set_description(f"Training epoch {epoch}/{args.epoch}")
#            if args.grav_pred or args.bounces_pred:
#                frames, gravs_or_bounces = batch # gravs = torch.Tensor([x_grav, y_grav, z_grav]), bounces = torch.Tensor([n_bounces])
#                frames, gravs_or_bounces = frames.float().to(args.device), gravs_or_bounces.to(args.device)
#                #import ipdb; ipdb.set_trace()
#                out = model_fwd(model, frames, args)
#                loss = criterion(out, gravs_or_bounces)
#                if args.bounces_pred:
#                    rounded_out = torch.round(out)
#                    gravs_or_bounces = torch.round(gravs_or_bounces)
#                    train_acc.append(torch.sum( rounded_out == gravs_or_bounces )/float(len(rounded_out)))
#                if args.grav_pred:
#                    rounded_out = (out*10).round()/10       # Rounding to 1 decimal place
#                    gravs_or_bounces = (gravs_or_bounces*10).round()/10# Same here
#                    train_acc.append(torch.sum( rounded_out == gravs_or_bounces )/float(len(rounded_out)))
#            else:
#                frames, gt_frames = batch
#                frames = frames.float().to(args.device)
#                gt_frames = gt_frames.float().to(args.device)
#                out = model_fwd(model, frames, args)
#                loss = criterion(out, gt_frames)
#                if args.loss == "SSIM":
#                    loss = 1 - loss
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#            train_loss.append(loss.item())
#        pbar.close()
#
#    # Validate
#    train_loss = sum(train_loss) / float(len(train_loss))#from train_corrects 
#    if args.bounces_pred or args.grav_pred:
#        train_acc = sum(train_acc) / float(len(train_acc))
#        valid_loss, valid_acc = validate(args, dset, model, criterion)
#    else:
#        valid_loss = validate(args, dset, model, criterion)
#    if args.bounces_pred or args.grav_pred:
#        return train_loss, valid_loss, train_acc, valid_acc # TODO Move to pytorch lightning, this is janky as hell
#    else:
#        return train_loss, valid_loss

##########
#def self_output(args, model, dset):
#    set_modes(dset, model, "self_out", args)
#    vis_loader = DataLoader(dset, batch_size=1, shuffle=args.shuffle)#, drop_last=True)
#    wandb_frames = []
#    wandb_metric_n_names = []
#    metrics = {
#        'PSNR':piq.psnr,
#        'LPIPS':piq.LPIPS(),
#        'SSIM':piq.ssim,
#        'MS_SSIM':piq.multi_scale_ssim,
#        'SL1':,
#        #'FID':piq.FID(),
#        #'FVD':tools.loss.FVD()
#    }
#
#    with tqdm(total=args.n_gifs) as pbar:
#        for ngif in range(args.n_gifs):
#            pbar.update(1)
#            pbar.set_description(f"Self output: {ngif}/{args.n_gifs}")
#            start_frames, gt_frames, vid_name = next(iter(self_out_loader))
#            start_frames = start_frames.float().to(args.device)
#            #gt_frames = gt_frames.float().to(args.device)
#            out = model_fwd(model, start_frames, args)
#            gif_frames = []
#            if args.self_output_n == -1:
#                self_output_n = gt_frames.shape[1]
#            else:
#                self_output_n = args.self_output_n
#            for itr in range(self_output_n):
#                start_frames = torch.cat([ start_frames[:,args.out_no:args.in_no] , out ], 1)
#                out = model_fwd(model, start_frames, args)
#                for n in range(args.out_no):
#                    gif_frames.append(out[0][n].cpu().detach().byte())
#                # Add the ground truth frame side by side to generated frame
#            gif_metrics = get_gif_metrics(gif_frames, gt_frames, metrics)
#            gif_frames = [ torch.cat( [torch.stack(gif_frames)[n_frm], gt_frames[0][n_frm]], 0) for n_frm in range(len(gif_frames)) ]
#            gif_save_path = os.path.join(args.results_dir, "%d.gif" % ngif)
#            imageio.mimsave(gif_save_path, gif_frames)
#            #args.plotter.gif_plot(args.jobname+" self_output"+str(ngif), gif_save_path)
#            wandb_frames.append(wandb.Video(gif_save_path))
#            gif_metrics['name'] = vid_name[0]
#            wandb_metric_n_names.append(gif_metrics)
#        pbar.close()
#    wandb.log({"self_output_gifs": wandb_frames}, commit=False)
#    wandb.log({"metrics":wandb_metric_n_names}, commit=False)
#    return wandb_metric_n_names


#def validate(args, dset, model, criterion):
#    set_modes(dset, model, "valid", args)
#    valid_loader = DataLoader(dset, batch_size=args.val_bsz, shuffle=args.shuffle)#, drop_last=True)
#    valid_loss = []
#    valid_acc = []
#    with tqdm(total=len(valid_loader)) as pbar:
#        for batch_idx, batch in enumerate(valid_loader):
#            pbar.update(1)
#            pbar.set_description(f"Validating...")
#            if args.grav_pred or args.bounces_pred:
#                frames, gravs_or_bounces = batch
#                frames, gravs_or_bounces = frames.float().to(args.device), gravs_or_bounces.to(args.device)
#                img = model_fwd(model, frames, args)
#                loss = criterion(img, gravs_or_bounces) # This is MSELoss
#                if args.bounces_pred:
#                    rounded_out = torch.round(img)
#                    gravs_or_bounces = torch.round(gravs_or_bounces)
#                    valid_acc.append(torch.sum(rounded_out == gravs_or_bounces)/float(len(rounded_out)))
#                if args.grav_pred:
#                    rounded_out = (img*10).round()/10# Rounding to 1 decimal place
#                    gravs_or_bounces = (gravs_or_bounces*10).round()/10
#                    valid_acc.append(torch.sum( rounded_out == gravs_or_bounces )/float(len(rounded_out)))
#            else:
#                frames, gt_frames = batch
#                frames = frames.float().to(args.device)
#                gt_frames = gt_frames.float().to(args.device)
#                img = model_fwd(model, frames, args)
#                loss = criterion(img, gt_frames)
#                if args.loss == "SSIM":
#                    loss = 1 - loss
#            valid_loss.append(loss.item())
#        pbar.close()
#    if args.bounces_pred or args.grav_pred:
#        return sum(valid_loss)/float(len(valid_loss)), sum(valid_acc)/float(len(valid_acc))
#    else:
#        return sum(valid_loss)/len(valid_loss)


