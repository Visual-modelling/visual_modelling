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

def visualise_imgs(args, dset, model,  n):
    """
    n is the number of images to visualise
    """
    dset.set_mode("valid")
    vis_loader = DataLoader(dset, batch_size=1, shuffle=True)
    # Image grid
    return_imgs = []
    for batch_idx, batch in enumerate(vis_loader):
        if batch_idx == n:
            print("n images visualised. Exiting.")
            return(return_imgs)
        else:
            frames, positions, gt_frames, gt_positions = batch
            frames_use, gt_frames_use = frames[0], gt_frames[0]

            # Convert old frames back to images and show them on the grid
            f, axarr = plt.subplots(args.in_no+args.out_no,2)
            for x in range(args.in_no):
                axarr[x,0].imshow(to_pil_image(frames_use[x]))
                axarr[x,1].imshow(to_pil_image(frames_use[x]))
            for y in range(args.out_no):
                axarr[args.in_no+y,0].imshow(to_pil_image(gt_frames_use[y]))

            frames = frames.squeeze(2).float().to(args.device)
            gt_frames = gt_frames.squeeze(2)#.to(args.device)
            out = model(frames).squeeze(2)
            out = out[0]
            out = torch.round(out).cpu().int()
            
            # Conert produce frame back into image and add to plots
            for y in range(args.out_no):
                axarr[args.in_no+y,1].imshow(to_pil_image(out[y]))

            # Calculate image comparison metrics
            plt.suptitle("Left = Ground Truth | Right = Predicted Final %d Frame(s)\n PSNR: %.3f, SSIM: %.3f, MS-SSIM: %.3f " 
                            % (args.out_no, 
                                PSNR(out.float(),gt_frames.squeeze(0).float(),1), 
                                ssim(out.unsqueeze(0).float(),gt_frames.float(),1), 
                                0))
            return_path = os.path.join(args.results_dir, "%d.png" % (batch_idx))
            plt.savefig(return_path)
            return_imgs.append(return_path)

                                # bugged right now ms_ssim(gt_frames.float(), out.unsqueeze(0).float(), 1)))


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
    model.load_state_dict(torch.load(args.checkpoint_path))
    #model.to(args.device)
    if args.visdom:
        args.plotter = VisdomLinePlotter(env_name=args.jobname)
    
    visualise_imgs(args, dset, model, args.vis_n)
