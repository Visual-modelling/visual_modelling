__author__ = "Jumperkables"

import os, sys, argparse, shutil
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VMDataset_v1
from models.UpDown2D import FCUp_Down
from visualiser import visualise_imgs 
import tools.utils as utils
import tools.radam as radam
import tools.loss as myloss
import torch.nn.functional as F
from tools.visdom_plotter import VisdomLinePlotter

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
        frames, positions, gt_frames, gt_positions = batch
        frames = frames.squeeze(2).float().to(args.device)
        gt_frames = gt_frames.squeeze(2).float().to(args.device)
        out = model(frames).squeeze(2)
        if args.loss == 'focal':
            loss = criterion(out, gt_frames)
        else:
            loss = criterion(out, gt_frames, reduction=args.reduction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        # Validate
        if(batch_idx % args.log_freq) == 0 and batch_idx != 0:
            train_loss = sum(train_loss) / float(len(train_loss))#from train_corrects            
            args.plotter.plot("%s loss" % (args.loss), "train", args.jobname, niter, train_loss)
            train_loss = []

            # Validation
            valid_loss = validate(args, valid_loader, model, criterion)
            args.plotter.plot(args.loss, "val", args.jobname, niter, valid_loss)
            
            # If this is the best run yet
            if valid_loss < previous_best_loss:
                previous_best_loss = valid_loss


                # Plot best accuracy so far in text box
                args.plotter.text_plot(args.jobname+" val", "Best %s val %.4f Iteration:%d" % (args.loss, previous_best_loss, niter))

                if args.save:
                    # Model save
                    torch.save(model.state_dict(), args.checkpoint_path)

                    # Save 5 example images & Plot one to visdom
                    img_paths = visualise_imgs(args, vis_loader, model, 5)
                    with Image.open(img_paths[0]) as plot_im:
                        plot_ret = transforms.ToTensor()(plot_im)
                    args.plotter.im_plot(args.jobname+" val", plot_ret) # Torch uint 0-255
                    

    return previous_best_loss


def validate(args, valid_loader, model, criterion):
    print("Validating")
    model.eval()
    valid_loss = []
    for batch_idx, batch in enumerate(valid_loader):
        frames, positions, gt_frames, gt_positions = batch
        frames = frames.squeeze(2).float().to(args.device)
        gt_frames = gt_frames.squeeze(2).float().to(args.device)
        img = model(frames)
        loss = criterion(img, gt_frames)
        valid_loss.append(loss.item())
    return sum(valid_loss)/len(valid_loss)

if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=2, help="number of epochs after no improvement before stopping")
    parser.add_argument("--log_freq", type=int, default=5000, help="iterations between each run of the validation set")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--raw_frame_rootdir", type=str, default=os.path.expanduser("~/"), help="The root directory containing 00000, 00001...")
    parser.add_argument("--extract_n_dset_file", action="store_true", help="activate this if you would like to extract your n_dset")
    parser.add_argument("--extracted_n_dset_savepathdir", type=str, default=os.path.expanduser("~/"), help="root directory of where you would like to save your n_dset.pickle")
    parser.add_argument("--dataset_path", type=str, default=os.path.expanduser("~/"))
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")
    parser.add_argument("--visdom", action="store_true", help="use a visdom ploter")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")
    """
    Please note, in_no + out_no must equal the size your dataloader is returning. Default is batches of 6 frames, 5 forward and 1 for ground truth
    """
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--repo_rootdir", type=str, default=os.path.expanduser("~/cnn_visual_modelling"), help="The rootdir of the cnn_visual_modelling repo")
    parser.add_argument("--save", action="store_true", help="Save models/validation things to checkpoint location")
    ####
    ##
    # Model Args
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--krnl_size", type=int, default=3, help="kernel sizes of the networks")
    parser.add_argument("--padding", type=int, default=1, help="Padding for kernel 0")
    parser.add_argument("--depth", type=int, default=2, help="depth of the updown")
    parser.add_argument("--channel_factor", type=int, default=64, help="channel scale factor for up down network")
    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "smooth_l1", "focal"], help="Loss function for the network")
    parser.add_argument("--reduce", action="store_true", help="reduction of loss function toggled")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")
    ####
    # Sorting arguements
    args = parser.parse_args()
    print(args)
    dset = VMDataset_v1(args)
    if args.save:
        results_dir = os.path.join(args.repo_rootdir, ".results", args.jobname )
        if(os.path.isdir(results_dir)):
            shutil.rmtree(results_dir)
            os.makedirs(results_dir)
        else:
            os.makedirs(results_dir)
        args.results_dir = results_dir
        args.checkpoint_path = os.path.join(args.results_dir, "model.pth")

    # Model info
    model = FCUp_Down(args)#.depth, args.in_no, args.out_no, args)
    args.device = torch.device("cuda:%d" % args.device if args.device>=0 else "cpu")
    model.to(args.device)
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad],
                                    lr=3e-4, weight_decay=1e-5)
    #criterion = loss.l1_loss
    # Losses
    if args.loss == "MSE":
        criterion = F.mse_loss#(reduction=args.reduction)#torch.nn.MSELoss(reduce=args.reduce, reduction=args.reduction).to(args.device)
    elif args.loss == "focal":
        criterion = myloss.FocalLoss(reduce=args.reduce, reduction=args.reduction).to(args.device) 
    elif args.loss == "smooth_l1":
        criterion = F.smooth_l1_loss#(reudction=args.reduction) #loss.SmoothL1Loss(reduce=args.reduce, reduction=args.reduction).to(args.device)
    else:
        raise Exception("Loss not implemented")
    if args.visdom:
        args.plotter = VisdomLinePlotter(env_name=args.jobname)
    if args.extract_n_dset_file:
        dset.save_dataset(args.in_no, args.out_no, args.extracted_n_dset_savepathdir)
        print("Extraction successful. Saved to:\n", args.extracted_n_dset_savepathdir+"/"+str(args.in_no+args.out_no)+"_dset.pickle")
        sys.exit()

    # Training loop
    early_stop_count = 0
    early_stop_flag = False
    best_loss = 10**20    # Mean average precision
    for epoch in range(args.epoch):
        if not early_stop_flag:
            epoch_best_loss = train(args, dset, model, optimizer, criterion, epoch, best_loss)
            if epoch_best_loss > best_loss:   # If the best loss for an epoch doesnt beat the current best
                early_stop_count += 1
                if early_stop_count >= args.early_stopping:
                    early_stop_flag = True
            else:
                best_loss = epoch_best_loss
        else:
            print_string = "Early stop on epoch %d/%d. Best %s %.3f at epoch %d" % (epoch+1, args.epoch, args.loss, best_loss, epoch+1-early_stop_count)
            print(print_string)
            args.plotter.text_plot(args.jobname+" epoch", print_string)
