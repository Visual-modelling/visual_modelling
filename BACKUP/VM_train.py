__author__ = "Jumperkables"

import os, sys, argparse, shutil, copy
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MMNIST, Dataset_from_raw
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

from tools.visdom_plotter import VisdomLinePlotter
import wandb
import piq

import imageio

def train(args, dset, model, optimizer, criterion, epoch, previous_best_loss):
    # Prepare validation loader
    #(dset, model, mode, args)
    set_modes(dset, model, "train", args)
    train_loader = DataLoader(dset, batch_size=args.bsz, shuffle=args.shuffle)#, drop_last=True)
    train_loss = []
    train_acc = []
    with tqdm(total=len(train_loader)) as pbar:
        for batch_idx, batch in enumerate(train_loader):
            pbar.update(1)
            pbar.set_description(f"Training epoch {epoch}/{args.epoch}")
            if args.grav_pred or args.bounces_pred:
                frames, gravs_or_bounces = batch # gravs = torch.Tensor([x_grav, y_grav, z_grav]), bounces = torch.Tensor([n_bounces])
                frames, gravs_or_bounces = frames.float().to(args.device), gravs_or_bounces.to(args.device)
                #import ipdb; ipdb.set_trace()
                out = model_fwd(model, frames, args)
                loss = criterion(out, gravs_or_bounces)
                if args.bounces_pred:
                    rounded_out = torch.round(out)
                    gravs_or_bounces = torch.round(gravs_or_bounces)
                    train_acc.append(torch.sum( rounded_out == gravs_or_bounces )/float(len(rounded_out)))
                if args.grav_pred:
                    rounded_out = (out*10).round()/10       # Rounding to 1 decimal place
                    gravs_or_bounces = (gravs_or_bounces*10).round()/10# Same here
                    train_acc.append(torch.sum( rounded_out == gravs_or_bounces )/float(len(rounded_out)))
            else:
                frames, gt_frames = batch
                frames = frames.float().to(args.device)
                gt_frames = gt_frames.float().to(args.device)
                out = model_fwd(model, frames, args)
                loss = criterion(out, gt_frames)
                if args.loss == "SSIM":
                    loss = 1 - loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        pbar.close()

    # Validate
    train_loss = sum(train_loss) / float(len(train_loss))#from train_corrects 
    if args.bounces_pred or args.grav_pred:
        train_acc = sum(train_acc) / float(len(train_acc))
        valid_loss, valid_acc = validate(args, dset, model, criterion)
    else:
        valid_loss = validate(args, dset, model, criterion)
    if args.bounces_pred or args.grav_pred:
        return train_loss, valid_loss, train_acc, valid_acc # TODO Move to pytorch lightning, this is janky as hell
    else:
        return train_loss, valid_loss

def get_gif_metrics(gif_frames, gt_frames, metrics):
    #import ipdb; ipdb.set_trace()
    gif_frames, gt_frames = torch.stack(gif_frames).unsqueeze(1), gt_frames[0].unsqueeze(1)
    metric_vals = {
        'PSNR':float(metrics['PSNR'](gif_frames, gt_frames, data_range=255., reduction="mean")),
        'LPIPS':float(metrics['LPIPS'](gif_frames.float(), gt_frames.float())),
        'SSIM':float(metrics['SSIM'](gif_frames, gt_frames, data_range=255.)),
        'MS_SSIM':-1. if (gif_frames.shape[2]<=161 and gif_frames.shape[3]<=161) else float(metrics['MS_SSIM'](gif_frames, gt_frames, data_range=255.)),
        #'FID':float(metrics['FID'](metrics['FID']._compute_feats(DataLoader(FID_dset(gif_frames.float()))), metrics['FID']._compute_feats(DataLoader(FID_dset(gt_frames.float()))))),
        #'FVD':metrics['FVD'](gif_frames[:utils.round_down(gif_frames.shape[0],16)], gt_frames[:utils.round_down(gt_frames.shape[0],16)] )
        #'FVD':metrics['FVD'](gif_frames, gt_frames)
    }
    return metric_vals

#### Wrapper classes for FID loss
class FID_dset(torch.utils.data.Dataset):
    def __init__(self, frames):
        self.frames = frames
    def __len__(self):
        return(self.frames.shape[0])
    def __getitem__(self, idx):
        return self.frames(idx)

#class FID_dloader(DataLoader):



##########
def self_output(args, model, dset):
    set_modes(dset, model, "self_out", args)
    vis_loader = DataLoader(dset, batch_size=1, shuffle=args.shuffle)#, drop_last=True)
    wandb_frames = []
    wandb_metric_n_names = []
    metrics = {
        'PSNR':piq.psnr,
        'LPIPS':piq.LPIPS(),
        'SSIM':piq.ssim,
        'MS_SSIM':piq.multi_scale_ssim,
        #'FID':piq.FID(),
        #'FVD':tools.loss.FVD()
    }

    with tqdm(total=args.n_gifs) as pbar:
        for ngif in range(args.n_gifs):
            pbar.update(1)
            pbar.set_description(f"Self output: {ngif}/{args.n_gifs}")
            start_frames, gt_frames, vid_name = next(iter(vis_loader))
            start_frames = start_frames.float().to(args.device)
            #gt_frames = gt_frames.float().to(args.device)
            out = model_fwd(model, start_frames, args)
            gif_frames = []
            if args.self_output_n == -1:
                self_output_n = gt_frames.shape[1]
            else:
                self_output_n = args.self_output_n
            for itr in range(self_output_n):
                start_frames = torch.cat([ start_frames[:,args.out_no:args.in_no] , out ], 1)
                out = model_fwd(model, start_frames, args)
                for n in range(args.out_no):
                    gif_frames.append(out[0][n].cpu().detach().byte())
                # Add the ground truth frame side by side to generated frame
            gif_metrics = get_gif_metrics(gif_frames, gt_frames, metrics)
            gif_frames = [ torch.cat( [torch.stack(gif_frames)[n_frm], gt_frames[0][n_frm]], 0) for n_frm in range(len(gif_frames)) ]
            gif_save_path = os.path.join(args.results_dir, "%d.gif" % ngif)
            imageio.mimsave(gif_save_path, gif_frames)
            #args.plotter.gif_plot(args.jobname+" self_output"+str(ngif), gif_save_path)
            wandb_frames.append(wandb.Video(gif_save_path))
            gif_metrics['name'] = vid_name[0]
            wandb_metric_n_names.append(gif_metrics)
        pbar.close()
    wandb.log({"self_output_gifs": wandb_frames}, commit=False)
    wandb.log({"metrics":wandb_metric_n_names}, commit=False)
    return wandb_metric_n_names


def validate(args, dset, model, criterion):
    set_modes(dset, model, "valid", args)
    valid_loader = DataLoader(dset, batch_size=args.val_bsz, shuffle=args.shuffle)#, drop_last=True)
    valid_loss = []
    valid_acc = []
    with tqdm(total=len(valid_loader)) as pbar:
        for batch_idx, batch in enumerate(valid_loader):
            pbar.update(1)
            pbar.set_description(f"Validating...")
            if args.grav_pred or args.bounces_pred:
                frames, gravs_or_bounces = batch
                frames, gravs_or_bounces = frames.float().to(args.device), gravs_or_bounces.to(args.device)
                img = model_fwd(model, frames, args)
                loss = criterion(img, gravs_or_bounces) # This is MSELoss
                if args.bounces_pred:
                    rounded_out = torch.round(img)
                    gravs_or_bounces = torch.round(gravs_or_bounces)
                    valid_acc.append(torch.sum(rounded_out == gravs_or_bounces)/float(len(rounded_out)))
                if args.grav_pred:
                    rounded_out = (img*10).round()/10# Rounding to 1 decimal place
                    gravs_or_bounces = (gravs_or_bounces*10).round()/10
                    valid_acc.append(torch.sum( rounded_out == gravs_or_bounces )/float(len(rounded_out)))
            else:
                frames, gt_frames = batch
                frames = frames.float().to(args.device)
                gt_frames = gt_frames.float().to(args.device)
                img = model_fwd(model, frames, args)
                loss = criterion(img, gt_frames)
                if args.loss == "SSIM":
                    loss = 1 - loss
            valid_loss.append(loss.item())
        pbar.close()
    if args.bounces_pred or args.grav_pred:
        return sum(valid_loss)/float(len(valid_loss)), sum(valid_acc)/float(len(valid_acc))
    else:
        return sum(valid_loss)/len(valid_loss)


def set_modes(dset, model, mode, args):
    if mode == "train":
        model.train()
    elif mode == "valid":
        model.eval()
    elif mode == "self_out":
        model.eval()
    if  1 < len(args.dataset):
        mixed_set_mode(dset, mode)
    else:
        dset.set_mode(mode)

def mixed_set_mode(dset, mode):
    for dst in dset.datasets:
        dst.set_mode(mode)


if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Run specific arguments")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=2, help="number of epochs after no improvement before stopping")
    """
    Guide to split_condition:
        'tv_ratio:4-1' : Simply split all videos into train:validation ratio of 4:1

    """
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--model_in_no", type=int, default=False, help="Use to assert model in_no regardless of dataset in_no")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--save", action="store_true", help="Save models/validation things to checkpoint location")
    parser.add_argument("--dset_sze", type=int, default=-1, help="Number of training samples from dataset")

    parser.add_argument_group("Dataset specific arguments")
    ############# To combine multiple datasets together, align the dataset and dataset path arguments
    parser.add_argument("--dataset", type=str, nargs="+", default="from_raw", choices=["mmnist", "from_raw"], help="Type of dataset")
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
    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "smooth_l1", "focal", "SSIM"], help="Loss function for the network")
    parser.add_argument("--reduce", action="store_true", help="reduction of loss function toggled")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")

    #####
    ## TRANSFORMER ARGS CAN GO HERE
    #####
    parser.add_argument_group("Transformer arguments")
    parser.add_argument("--transformer_example_arg", type=str, default="Start here", help="Feel free to start here")

    parser.add_argument_group("Logging arguments")
    parser.add_argument("--visdom", action="store_true", help="use a visdom ploter")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")
    parser.add_argument("--self_output", action="store_true", help="Run self output on specified model")
    parser.add_argument("--self_output_n", type=int, default=-1, help="Number of frames to run self output on")
    parser.add_argument("--model_path", type=str, default="", help="path of saved model")
    parser.add_argument("--gif_path", type=str, default=os.path.expanduser("~/cnn_visual_modelling/.results/self_out.gif"), help="path to save the gif")
    parser.add_argument("--n_gifs", type=int, default=10, help="number of gifs to save")
    
    ####
    # Sorting arguements
    args = parser.parse_args()
    print(args)

    # If this is a classification task, make sure its only one of them
    task_count = (args.segmentation)+(args.grav_pred)+(args.bounces_pred)
    assert task_count<=1, f"You can only run one of gravity prediction, segmentation, or bounce prediction"

    if args.segmentation:
        assert args.in_no == args.out_no == 1, f"Image segmentation is defined on 1 input and 1 output image. Not {args.in_no} and {args.out_no}"

    assert len(args.dataset) == len(args.dataset_path), f"Number of specified dataset paths and dataset types should be equal"
    dataset_swtich = {
        "from_raw" : Dataset_from_raw,
        "mmnist" : MMNIST
    }
    if len(args.dataset) > 1:  # If multiple dataset
        dataset_list = args.dataset_path
        dataset_list = [dataset_swtich[args.dataset[i]](args.dataset_path[i], args) for i in range(len(args.dataset))]
        dset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        dset = dataset_swtich[args.dataset[0]](args.dataset_path[0], args)

    if args.save:
        repo_rootdir = os.path.dirname(os.path.realpath(sys.argv[0]))
        results_dir = os.path.join(repo_rootdir, ".results", args.jobname )
        if (os.path.isdir(results_dir)):
            if args.model_path == "": #i.e. IF LOAD PATH IS DEFAULT/UNSET
                raise FileExistsError(f"Before you can re-run, please manually delete {results_dir}")
            else:
                pass
                #shutil.rmtree(results_dir)
                #os.makedirs(results_dir)
        else:
            if args.model_path == "":
                os.makedirs(results_dir)
            else:
                raise FileNotFoundError(f"You want to load a model, but {results_dir} does not exist")
        args.results_dir = results_dir
        args.checkpoint_path = os.path.join(args.results_dir, "model.pth")

    # Model info
    if args.model == "UpDown3D":
        model = FCUp_Down3D(args)
    elif args.model == "UpDown2D":
        if args.segmentation:
            model = FCUp_Down2D_2_Segmentation(args, args.model_path, load_mode="pad")
        elif args.grav_pred:
            model = FCUp_Down2D_2_FCPred(args, args.model_path, mode="grav_pred")
        elif args.bounces_pred:
            model = FCUp_Down2D_2_FCPred(args, args.model_path, mode="bounces_pred")
        else:
            model = FCUp_Down2D(args)#.depth, args.in_no, args.out_no, args)
    elif args.model == "transformer":
        import ipdb; ipdb.set_trace()
        print("Its all up to you from here <3 ")
        print("I'm not intimately familiar with Deans code. Its changed since I've used it and hes said he'll push extra changes.")
        print("You will have to go into transformer.py and intimately understand and change it. I dont know how correct his implementation is or anything.")
        print("I have given you the tools and explanations to get started with all this. Keep an eye on my master branche's readme for details on extra functionality, like how to concatenate multiple datasets etc...")
        print("The only change I've made to either transformer object is passing my 'args' into it and binding. Up to you how you choose to use them")
        print("Ask me if anything too unclear. Good luck friends <3")
        model = VMTransformer2(args) 
        import sys; sys.exit()
    else:
        raise Exception("Model: %s not implemented" % (args.model))

    args.device = torch.device("cuda:%d" % args.device if args.device>=0 else "cpu")
    model.to(args.device)
    
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
    # Losses
    if args.loss == "MSE":
        criterion = torch.nn.MSELoss(reduce=args.reduce, reduction=args.reduction).to(args.device)
    elif args.loss == "focal":
        criterion = tools.loss.FocalLoss(reduce=args.reduce, reduction=args.reduction).to(args.device) 
    elif args.loss == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss(reduce=args.reduce, reduction=args.reduction).to(args.device)
    elif args.loss == "SSIM":
        criterion = tools.loss.ssim(data_range=255, size_average=True, channel=1).to(args.device)
    else:
        raise Exception("Loss not implemented")
    # TODO make this overwrite of criterion more elegant
    if args.grav_pred:
        criterion = torch.nn.MSELoss().to(args.device)
    if args.bounces_pred:
        criterion = torch.nn.MSELoss().to(args.device)
    if args.visdom:
        #args.plotter = VisdomLinePlotter(env_name=args.jobname)
        wandb.init(project="visual-modelling", entity="visual-modelling", name=args.jobname, resume="allow")
        #wandb.config.update(args, allow_val_change=True)

    # Training loop
    early_stop_count = 0
    early_stop_flag = False
    best_loss = 10**20    # Mean average precision
    if args.epoch == 0:
        validate(args, dset, model, criterion)
    else:
        print("Training Start")
    for epoch in range(args.epoch):
        if not early_stop_flag:
            if args.bounces_pred or args.grav_pred:
                train_loss, valid_loss, train_acc, valid_acc = train(args, dset, model, optimizer, criterion, epoch, best_loss)
            else:
                train_loss, valid_loss = train(args, dset, model, optimizer, criterion, epoch, best_loss)
            if valid_loss > best_loss:  # No improvement
                early_stop_count += 1
                if early_stop_count >= args.early_stopping:
                    early_stop_flag = True
            else:                       # New Best Epoch
                early_stop_count  = 0
                best_loss = valid_loss
                if args.visdom:
                    pass
                    #args.plotter.text_plot(args.jobname+" val", "Best %s val %.4f Iteration:%d" % (args.loss, best_loss, epoch))
                if args.save:
                    torch.save(model.state_dict(), args.checkpoint_path)
                    if task_count == 0: # If this isnt segmentation/gravity prediction etc..
                        metrics = self_output(args, model, dset)
                        psnr_metric = utils.avg_list([ empl["PSNR"] for empl in metrics ])
                        ssim_metric = utils.avg_list([ empl["SSIM"] for empl in metrics ])
                        msssim_metric = utils.avg_list([ empl["MS_SSIM"] for empl in metrics ])
                        lpips_metric = utils.avg_list([ empl["LPIPS"] for empl in metrics ])
                        # Printing and logging either way
                        print(f"Epoch {epoch}/{args.epoch}", f"Train Loss {train_loss:.3f} |", f"Val Loss {valid_loss:.3f} |", f"Early Stop Flag {early_stop_count}/{args.early_stopping}", f"PSNR {psnr_metric:.3f}", f"SSIM {ssim_metric:.3f}", f"MS-SSIM {msssim_metric:.3f}", f"LPIPS {lpips_metric:.3f}\n")
                    model.train()
                else:
                    # Printing and logging either way
                    print(f"Epoch {epoch}/{args.epoch}", f"Train Loss {train_loss:.3f} |", f"Val Loss {valid_loss:.3f} |", f"Early Stop Flag {early_stop_count}/{args.early_stopping}\n")
            if args.visdom:
                if args.save:
                    if task_count == 0: # If this isnt segmentation/gravity prediction etc..
                        wandb.log({'PSNR' : psnr_metric}, commit=False)
                        wandb.log({'SSIM' : ssim_metric}, commit=False)
                        wandb.log({'MS-SSIM' : msssim_metric}, commit=False)
                        wandb.log({'LPIPS' : lpips_metric}, commit=False)
                wandb.log({'val_loss' : best_loss})
                wandb.log({'train_loss':train_loss})
                if args.bounces_pred or args.grav_pred:
                    wandb.log({'train_acc' : train_acc})
                    wandb.log({'valid_acc' : valid_acc})
                #args.plotter.text_plot(args.jobname+" epoch", f"Train Loss {train_loss:.3f} | Val Loss {valid_loss:.3f} | Early Stop Count {early_stop_count}")
                #args.plotter.plot(args.loss, "val", "val "+args.jobname, epoch, valid_loss)
                #args.plotter.plot(args.loss, "train", "train "+args.jobname, epoch, train_loss)

        else:
            print_string = "Early stop on epoch %d/%d. Best %s %.3f at epoch %d" % (epoch+1, args.epoch, args.loss, best_loss, epoch+1-early_stop_count)
            print(print_string)
            #args.plotter.text_plot(args.jobname+" epoch", print_string)
            sys.exit()
