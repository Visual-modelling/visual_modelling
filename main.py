__author__ = "Jumperkables"

import os, sys, argparse, shutil
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VMDataset_v1, MMNIST
from models.UpDown2D import FCUp_Down2D
from models.UpDown3D import FCUp_Down3D
from models.transformer import VMTransformer, VMTransformer2 
from visualiser import visualise_imgs 
import tools.utils as utils
from tools.utils import model_fwd
import tools.radam as radam
import tools.loss
import torch.nn.functional as F
from tools.visdom_plotter import VisdomLinePlotter
import imageio

def train(args, dset, model, optimizer, criterion, epoch, previous_best_loss):
    dset.set_mode("val")
    vis_loader = DataLoader(dset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(dset, batch_size=args.val_bsz, shuffle=True)
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=args.bsz, shuffle=True)
    train_loss = []
    for batch_idx, batch in enumerate(train_loader):
        #niter = epoch * len(train_loader) + batch_idx
        #print(niter, epoch, batch_idx)
        niter = round(epoch + (batch_idx/len(train_loader)), 3) # Changed niter to be epochs instead of iterations
        if args.dataset == "hudsons":
            frames, positions, gt_frames, gt_positions = batch
        elif args.dataset == "mmnist":
            frames, gt_frames = batch
            frames = frames.squeeze(2)
            gt_frames = gt_frames.squeeze(2)
        else:
            raise Exception(f"{args.dataset} is not implemented.")
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

    # Validate
    #print("Epoch %d done" % epoch)
    train_loss = sum(train_loss) / float(len(train_loss))#from train_corrects            
    args.plotter.plot("%s loss" % (args.loss), "train", "train "+args.jobname, niter, train_loss)
    #train_loss = []

    # Validation
    valid_loss = validate(args, valid_loader, model, criterion)
    args.plotter.plot(args.loss, "val", "val "+args.jobname, niter, valid_loss)
    
    # If this is the best run yet
    if valid_loss < previous_best_loss:
        previous_best_loss = valid_loss


        # Plot best accuracy so far in text box
        args.plotter.text_plot(args.jobname+" val", "Best %s val %.4f Iteration:%d" % (args.loss, previous_best_loss, niter))

        if args.save:
            # Model save
            torch.save(model.state_dict(), args.checkpoint_path)

            # Save 5 example images & Plot one to visdom
            visualise_imgs(args, vis_loader, model, 5)
            if args.self_output:
                self_output(args, model, vis_loader)
            model.train()

    return train_loss, previous_best_loss


def self_output(args, model, vis_loader):
    model.eval()
    print("self output commencing...")
    for ngif in range(args.n_gifs):
        if args.dataset == "hudsons":
            frames, positions, gt_frames, gt_positions = next(iter(vis_loader))
        elif args.dataset == "mmnist":
            frames, gt_frames = next(iter(vis_loader))
            frames = frames.squeeze(2)
            gt_frames = gt_frames.squeeze(2)
        else:
            raise Exception(f"{args.dataset} is not implemented.")

        frames = frames.float().to(args.device)
        gt_frames = gt_frames.float().to(args.device)
        out = model_fwd(model, frames, args)
        gif_frames = []
        for itr in range(args.self_output_n):
            frames = torch.cat([ frames[:,args.out_no:args.in_no] , out ], 1)
            out = model_fwd(model, frames, args)
            for n in range(args.out_no):
                gif_frames.append(out[0][n].cpu().detach().byte())
        gif_save_path = os.path.join(args.results_dir, "%d.gif" % ngif) 
        imageio.mimsave(gif_save_path, gif_frames)
        args.plotter.gif_plot(args.jobname+" self_output"+str(ngif), gif_save_path)
    print("self output finished!")  


def validate(args, valid_loader, model, criterion):
    print("Validating")
    model.eval()
    valid_loss = []
    for batch_idx, batch in enumerate(valid_loader):
        if args.dataset == "hudsons":
            frames, positions, gt_frames, gt_positions = batch
        elif args.dataset == "mmnist":
            frames, gt_frames = batch
            frames = frames.squeeze(2)
            gt_frames = gt_frames.squeeze(2)
        else:
            raise Exception(f"{args.dataset} is not implemented.")

        frames = frames.float().to(args.device)
        gt_frames = gt_frames.float().to(args.device)
        img = model_fwd(model, frames, args)
        loss = criterion(img, gt_frames)
        if args.loss == "SSIM":
            loss = 1 - loss
        valid_loss.append(loss.item())
    return sum(valid_loss)/len(valid_loss)







if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=2, help="number of epochs after no improvement before stopping")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--dataset_path", type=str, default=os.path.expanduser("~/"))
    parser.add_argument("--dataset", type=str, default="hudsons", choices=["hudsons","mmnist"])
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")
    parser.add_argument("--visdom", action="store_true", help="use a visdom ploter")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")
    """
    Please note, in_no + out_no must equal the size your dataloader is returning. Default is batches of 6 frames, 5 forward and 1 for ground truth
    """
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--save", action="store_true", help="Save models/validation things to checkpoint location")
    parser.add_argument("--extract_dset", action="store_true", help="activate this if you would like to extract your n_dset")

    ####
    ##
    # Model Args
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

    # Self input
    parser.add_argument("--self_output", action="store_true", help="Run self output on specified model")
    parser.add_argument("--self_output_n", type=int, default=100, help="Number of frames to run self output on")
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/cnn_visual_modelling/model.pth"), help="path of saved model")
    parser.add_argument("--gif_path", type=str, default=os.path.expanduser("~/cnn_visual_modelling/.results/self_out.gif"), help="path to save the gif")
    parser.add_argument("--n_gifs", type=int, default=10, help="number of gifs to save")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "transformer"], help="Type of model to run")

    ####
    # Sorting arguements
    args = parser.parse_args()
    print(args)
    if args.dataset == "hudsons":
        dset = VMDataset_v1(args)
    elif args.dataset == "mmnist":
        dset = MMNIST(args)
    else:
        raise Exception(f"{args.dataset} dataset not implemented")

    if args.save:
        repo_rootdir = os.path.dirname(os.path.realpath(sys.argv[0]))
        results_dir = os.path.join(repo_rootdir, ".results", args.jobname )
        if(os.path.isdir(results_dir)):
            shutil.rmtree(results_dir)
            os.makedirs(results_dir)
        else:
            os.makedirs(results_dir)
        args.results_dir = results_dir
        args.checkpoint_path = os.path.join(args.results_dir, "model.pth")


    # Model info
    if args.model == "UpDown3D":
        model = FCUp_Down3D(args)
    elif args.model == "UpDown2D":
        model = FCUp_Down2D(args)#.depth, args.in_no, args.out_no, args)
    elif args.model == "transformer":
        model = VMTransformer()
    else:
        raise Exception("Model: %s not implemented" % (args.model))

    args.device = torch.device("cuda:%d" % args.device if args.device>=0 else "cpu")
    model.to(args.device)
    
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad],
                                    lr=3e-4, weight_decay=1e-5)
    #criterion = loss.l1_loss
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
    if args.visdom:
        args.plotter = VisdomLinePlotter(env_name=args.jobname)

    # Training loop
    early_stop_count = 0
    early_stop_flag = False
    best_loss = 10**20    # Mean average precision
    print("Training Start")
    for epoch in range(args.epoch):
        if not early_stop_flag:
            train_loss, epoch_best_loss = train(args, dset, model, optimizer, criterion, epoch, best_loss)
            if epoch_best_loss > best_loss:   # If the best loss for an epoch doesnt beat the current best
                early_stop_count += 1
                if early_stop_count >= args.early_stopping:
                    early_stop_flag = True
            else:
                early_stop_count  = 0
                best_loss = epoch_best_loss
            print(f"Train Loss {train_loss:.3f} |",
                f"Val Loss {epoch_best_loss:.3f} |",
                f"Early Stop Flag {early_stop_count}")
            args.plotter.text_plot(args.jobname+" epoch", f"Train Loss {train_loss:.3f} | Val Loss {epoch_best_loss:.3f} | Early Stop Count {early_stop_count}")
        else:
            print_string = "Early stop on epoch %d/%d. Best %s %.3f at epoch %d" % (epoch+1, args.epoch, args.loss, best_loss, epoch+1-early_stop_count)
            print(print_string)
            args.plotter.text_plot(args.jobname+" epoch", print_string)
