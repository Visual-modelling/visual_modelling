__author__ = "Jumperkables"

import os, sys, argparse
import torch
from torch.utils.data import DataLoader

from dataset import VMDataset_v1
from models.FC3D import FC3D_1_0
import tools.utils as utils
import tools.radam as radam
import tools.loss as loss
from tools.visdom_plotter import VisdomLinePlotter

def train(args, dset, model, optimizer, criterion, epoch, previous_best_mse):
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=args.bsz, shuffle=True)
    train_loss = []
    for batch_idx, batch in enumerate(train_loader):
        niter = epoch * len(train_loader) + batch_idx
        print(niter, epoch, batch_idx)
        frames, positions, gt_frames, gt_positions = batch
        frames = frames.squeeze(2).float().to(args.device)
        gt_frames = gt_frames.squeeze(2).float().to(args.device)
        out = model(frames).squeeze(2)
        loss = criterion(out, gt_frames)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        # Validate
        if(batch_idx % args.log_freq) == 0 and batch_idx != 0:
            niter = epoch * len(train_loader) + batch_idx
            train_loss = sum(train_loss) / float(len(train_loss))#from train_corrects            
            args.plotter.plot("MSE loss", "train", args.jobname, niter, train_loss)
            train_loss = []

            # Validation
            valid_mse = validate(args, dset, model)
            args.plotter.plot("MSE", "val", args.jobname, niter, valid_mse)
            
            # If this is the best run yet
            if valid_mse < previous_best_mse:
                previous_best_mse = valid_mse

                # Plot best accuracy so far in text box
                args.plotter.text_plot(args.jobname+" val", "Best MSE val %.4f Iteration:%d" % (previous_best_mse, niter))

                # Model save
                torch.save(model.state_dict(), args.checkpoint_path)
    return previous_best_mse


def validate(args, dset, model):
    print("Validating")
    dset.set_mode("val")
    model.eval()
    valid_loader = DataLoader(dset, batch_size=args.val_bsz, shuffle=True)
    valid_mse = []
    mse_obj = torch.nn.MSELoss().to(args.device)
    for batch_idx, batch in enumerate(valid_loader):
        frames, positions, gt_frames, gt_positions = batch
        frames = frames.squeeze(2).float().to(args.device)
        gt_frames = gt_frames.squeeze(2).float().to(args.device)
        img = model(frames)
        mse = mse_obj(img, gt_frames)
        valid_mse.append(mse.item())
    return sum(valid_mse)/len(valid_mse)

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
    parser.add_argument("--checkpoint_path", type=str, default=os.path.expanduser("~/model.pth"), help="where to save best model")
    """
    Please note, in_no + out_no must equal the size your dataloader is returning. Default is batches of 6 frames, 5 forward and 1 for ground truth
    """
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    args = parser.parse_args()
    print(args)
    dset = VMDataset_v1(args)
    model = FC3D_1_0(args)
    model.to(args.device)
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad],
                                    lr=3e-4, weight_decay=1e-5)
    #criterion = loss.l1_loss
    criterion = torch.nn.MSELoss().to(args.device)
    if args.visdom:
        args.plotter = VisdomLinePlotter(env_name=args.jobname)
    if args.extract_n_dset_file:
        dset.save_dataset(args.in_no, args.out_no, args.extracted_n_dset_savepathdir)
        print("Extraction successful. Saved to:\n", args.extracted_n_dset_savepathdir+"/"+str(args.in_no+args.out_no)+"_dset.pickle")
        sys.exit()

    # Training loop
    early_stop_count = 0
    early_stop_flag = False
    best_mse = 0    # Mean average precision
    for epoch in range(args.epoch):
        if not early_stop_flag:
            epoch_best_mse = train(args, dset, model, optimizer, criterion, epoch, best_mse)
            if epoch_best_mse > best_mse:   # If the best Mse for an epoch doesnt beat the current best
                early_stop_count += 1
                if early_stop_count >= args.early_stopping:
                    early_stop_flag = True
            else:
                best_mse = epoch_best_mse
        else:
            print_string = "Early stop on epoch %d/%d. Best MSE %.3f at epoch %d" % (epoch+1, args.epoch, best_mse, epoch+1-early_stop_count)
            print(print_string)
            args.plotter.text_plot(args.jobname+" epoch", print_string)
