import os, sys
import torch
import argparse
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from models.UpDown2D import FCUp_Down2D_2_MNIST
import tools.radam as radam
import numpy as np
import tqdm
from tqdm import tqdm
from time import sleep

import wandb

def test_MNIST(model, args, bsz=1):
    #model.eval()
    dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
    test_loader = DataLoader(dset, batch_size=bsz)
    correct, total = 0, 0
    for batch in test_loader:
        img, label = batch
        img = (255*img).long().float()
        img, label = img.to(args.device), label.to(args.device)
        out = model(img)
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    acc = 100*correct/total 
    return acc, f"MNIST accuracy: {acc:.2f}%"


def train_MNIST(model, args, epochs=30, bsz=16):
    """
    Train on MNIST
    """
    model.train()
    dset = MNIST(train=True, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
    train_loader = DataLoader(dset, batch_size=bsz)
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
    early_stop_count = 0
    best_epoch = 0
    for epoch in range(epochs):
        train_loss = []
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                pbar.set_description(f"Training epoch {epoch}/{epochs}")
                pbar.update(1)
                #sleep(0.001)
                img, label = batch
                img = (255*img).long().float()
                img, label = img.to(args.device), label.to(args.device)
                out = model(img)
    
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            pbar.close()
        print("Validating...")
        train_loss = sum(train_loss)/len(train_loss)
        val_acc, _ = test_MNIST(model, args)
        model.train()
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stopping:
                return best_acc, f"Early Stopping @ {epoch} epochs. Best accuracy: {best_acc:.3f}% was at epoch {best_epoch}"

        print(f"Epoch:{epoch}/{epochs}, Loss:{train_loss:.5f}, Val Acc:{val_acc:.3f}%, Early Stop:{early_stop_count}/{args.early_stopping}")
        if args.visdom:
            wandb.log({'Train Loss' : train_loss})
            wandb.log({"Valid Accuracy": val_acc})
    return best_acc, f"{epochs} Epochs (full). Best Accuracy: {best_acc:.3f}% {'was at final epoch' if early_stop_count == 0 else 'was at epoch '+str(best_epoch)}"



if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Run specific arguments")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=2, help="number of epochs after no improvement before stopping")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--save", action="store_true", help="Save models/validation things to checkpoint location")

    parser.add_argument_group("Model specific arguments")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "transformer"], help="Type of model to run")
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/cnn_visual_modelling/model.pth"), help="path of saved model")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--krnl_size", type=int, default=3, help="Height and width kernel size")
    parser.add_argument("--krnl_size_t", type=int, default=3, help="Temporal kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Height and width Padding")
    parser.add_argument("--padding_t", type=int, default=1, help="Temporal Padding")
    parser.add_argument("--depth", type=int, default=2, help="depth of the updown")
    parser.add_argument("--channel_factor", type=int, default=64, help="channel scale factor for up down network")
    #parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "smooth_l1", "focal", "SSIM"], help="Loss function for the network")
    #parser.add_argument("--reduce", action="store_true", help="reduction of loss function toggled")
    #parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")

    parser.add_argument_group("Logging arguments")
    parser.add_argument("--visdom", action="store_true", help="use a visdom ploter")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")
    args = parser.parse_args()
    print(args)
    model = FCUp_Down2D_2_MNIST(args, load_path=args.model_path)
    model.to(args.device)
    if args.visdom:
        #args.plotter = VisdomLinePlotter(env_name=args.jobname)
        wandb.init(project="visual-modelling", entity="visual-modelling", name=args.jobname)
        wandb.config.update(args)

    # Training (includes validation after each epoch and early stopping)
    if args.epoch > 0:        
        best_acc, return_string = train_MNIST(model, args, bsz=args.bsz, epochs=args.epochs)

    # Only Testing
    if args.epoch == 0:
        best_acc, _ = test_MNIST(model, args, epochs=args.epoch, bsz=args.val_bsz)
        return_string = f"Validation Only: Accuracy: {best_acc:.2f}%"
    print(return_string) 
    if args.visdom:
        wandb.log({"Run Info" : return_string})
        wandb.log({'Best Accuracy' : best_acc})


