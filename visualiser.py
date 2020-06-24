_author__ = "Jumperkables"

import os, sys, argparse
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from dataset import VMDataset_v1
from tools.visdom_plotter import VisdomLinePlotter
from tools.loss import ssim, ms_ssim, PSNR
import numpy as np
import imageio

from tools.utils import img_merge, model_fwd

def visualise_imgs(args, vis_loader, model,  n):
    """
    n is the number of images to visualise
    """
    # Image grid
    return_imgs = []
    for batch_idx, batch in enumerate(vis_loader):
        if batch_idx == n:
            print("n images visualised. Exiting.")
            return(return_imgs)
        else:
            if args.dataset == "hudsons":
                frames, positions, gt_frames, gt_positions = batch
            elif args.dataset == "mmnist":
                frames, gt_frames = batch
                frames, gt_frames = frames.squeeze(2), gt_frames.squeeze(2)
            else:
                raise Exception(f"{args.dataset} is not implemented.")

           
            frames_use, gt_frames_use = frames[0], gt_frames[0]
            frames = frames.squeeze(2).float().to(args.device)
            gt_frames = gt_frames.squeeze(2)#.to(args.device)
            out = model_fwd(model, frames, args)
            out = out.squeeze(2)
            out = out[0]
            out = torch.round(out).cpu().int()

            # Create a gif from groundtruth vs generated
            gif_frames = [] 
            for x in range(args.in_no):    
                gif_frames.append(img_merge([frames_use[x]]*2, "greyscale", "horizontal"))
            for y in range(args.out_no):
                gif_frames.append(img_merge([gt_frames_use[y], out[y]], "greyscale", "horizontal"))
            imageio.mimsave(os.path.join(args.results_dir, "gt_vs_out.gif"), gif_frames)
            args.plotter.gif_plot(args.jobname+" gt_vs_out", os.path.join(args.results_dir, "gt_vs_out.gif"))


if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.expanduser("~/"))
    parser.add_argument("--visdom", action="store_true", help="use a visdom ploter")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")
    parser.add_argument("--checkpoint_path", type=str, default=os.path.expanduser("~/model.pth"), help="where to save best model")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--extract_n_dset_file", action="store_true", help="activate this if you would like to extract your n_dset")
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")
    """
    Please note, in_no + out_no must equal the size your dataloader is returning. Default is batches of 6 frames, 5 forward and 1 for ground truth
    """
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--vis_n", type=int, default=1, help="number of image sets to visualise")
    args = parser.parse_args()
    print(args)
    dset = VMDataset_v1(args)
    model = FC3D_1_0(args)
    args.checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.checkpoint_path)
    model.load_state_dict(torch.load(args.checkpoint_path))
    #model.to(args.device)
    if args.visdom:
        args.plotter = VisdomLinePlotter(env_name=args.jobname)
    
    visualise_imgs(args, dset, model, args.vis_n)
