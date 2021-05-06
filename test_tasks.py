import os, sys
import torch
import argparse

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from dataset import Dataset_from_raw
from models.UpDown2D import FCUp_Down2D_2_MNIST
import tools.radam as radam
import tools.loss
import tools.utils as utils

import numpy as np
import tqdm
from tqdm import tqdm
from time import sleep
import pandas as pd

import wandb

################################
# Utility Functions
################################
def HDMB_create_labels():
    HDMB51_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/testTrainMulti_7030_splits")
    HDMB51_labels = os.listdir(HDMB51_labels_path)
    train = {}
    test = {}
    other = {}
    for txtfile in HDMB51_labels:
        fname = txtfile.split(".")[0]
        ctgry, _ = fname.split("_test_split")
        #ctrgy, splt = fname.split("_test_split")
        assert len(fname.split("_test_split")) == 2, f"{txtfile} breaks this formatting"
        #splt = splt.split(".")[0]
        
        labs = open(f"{HDMB51_labels_path}/{txtfile}", "r").readlines()
        for ele in labs:
            vid, train_or_test =  ele.split()
            vid = vid.split(".")[0]                
            if train_or_test == "0":
                other[vid] = ctgry
            elif train_or_test == "1":
                train[vid] = ctgry
            elif train_or_test == "2":
                test[vid] = ctgry
            else:
                print(f"Should not have happend: Test-or-test ID:{train_or_test}")
    sorted_lab_type = sorted(set(train.values()))
    class2id = { lab:idx for idx, lab in enumerate(sorted_lab_type)}
    id2class = { idx:lab for idx, lab in enumerate(sorted_lab_type)}
    train   = ({ vid:class2id[lab] for vid, lab in train.items()}, class2id, id2class)
    test    = ({ vid:class2id[lab] for vid, lab in test.items()}, class2id, id2class)
    other   = ({ vid:class2id[lab] for vid, lab in other.items()}, class2id, id2class)
    utils.save_pickle(train, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))
    utils.save_pickle(test , os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/test_labels.pickle"))
    utils.save_pickle(other, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/other_labels.pickle"))






################################
# Main Functions
################################
# MNIST
def test_MNIST(model, args, bsz=1):
    model.eval()
    dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
    test_loader = DataLoader(dset, batch_size=bsz)
    correct, total = 0, 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        img, label = batch
        img = (255*img).long().float()
        #if args.load_mode == "pad":
        #    new_label = torch.ones(img.shape)
        #    for x in range(label.shape[0]):
        #        new_label[x] = new_label[x]*(label[x]*10) # Image of 0s is equal to class 0, image of 10s to class 1, 20s to class 2 etc..
        #    label = new_label
        img, label = img.to(args.device), label.to(args.device)
        out = model(img)
        if args.load_mode == "pad":
            predicted = torch.mean(out, [1,2,3])    # Average across all outputs and divide by 10 to determine predicted class
            predicted = torch.round(predicted * 0.1).long()
            
        elif args.load_mode == "replace":  
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
    if args.load_mode == "replace":
        criterion = nn.CrossEntropyLoss()
    elif args.load_mode == "pad":
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
                if args.load_mode == "pad":
                    new_label = torch.ones(img.shape)
                    for x in range(label.shape[0]):
                        new_label[x] = new_label[x]*(label[x]*10) # Image of 0s is equal to class 0, image of 10s to class 1, 20s to class 2 etc..
                    label = new_label
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


# HDMB
def test_HDMB_51(model, args, bsz=1):
    model.eval()
    args.dset_sze = -1
    #dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
    test_loader = DataLoader(dset, batch_size=bsz)
    correct, total = 0, 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        img, label = batch
        img = (255*img).long().float()
        #if args.load_mode == "pad":
        #    new_label = torch.ones(img.shape)
        #    for x in range(label.shape[0]):
        #        new_label[x] = new_label[x]*(label[x]*10) # Image of 0s is equal to class 0, image of 10s to class 1, 20s to class 2 etc..
        #    label = new_label
        img, label = img.to(args.device), label.to(args.device)
        out = model(img)
        #import ipdb; ipdb.set_trace()
        if args.load_mode == "pad":
            predicted = torch.mean(out, [1,2,3])    # Average across all outputs and divide by 10 to determine predicted class
            predicted = torch.round(predicted * 0.1).long()
            
        elif args.load_mode == "replace":  
            _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    
    acc = 100*correct/total 
    return acc, f"MNIST accuracy: {acc:.2f}%"



def train_HDMB_51(model, args, epochs=30, bsz=16):
    """
    Train on HDMB-51
    """
    #import ipdb; ipdb.set_trace()
    model.train()
    args.dset_sze = -1
    dset = Dataset_from_raw(args.dataset_path[0], args)
    train_loader = DataLoader(dset, batch_size=bsz)
    best_acc = 0
    if args.load_mode == "replace":
        criterion = nn.CrossEntropyLoss()
    elif args.load_mode == "pad":
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
                #import ipdb; ipdb.set_trace()
                img = (255*img).long().float()
                if args.load_mode == "pad":
                    new_label = torch.ones(img.shape)
                    for x in range(label.shape[0]):
                        new_label[x] = new_label[x]*(label[x]*10) # Image of 0s is equal to class 0, image of 10s to class 1, 20s to class 2 etc..
                    label = new_label
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
        val_acc, _ = test_HDMB_51(model, args)
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
    parser.add_argument("--TASK", type=str, choices=["mnist", "mocap", "hdmb51", "grav", ], help="Which task, classification or otherwise, to apply")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=2, help="number of epochs after no improvement before stopping")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--model_in_no", type=int, default=False, help="Use to assert model in_no regardless of dataset in_no")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--save", action="store_true", help="Save models/validation things to checkpoint location")
    parser.add_argument("--segmentation", action="store_true", help="Create a dataset for image segmentation. Segmentation masks for images in video clips should be named the same as the original image and stored in a subdirectory of the clip 'mask'")

    parser.add_argument_group("Dataset specific arguments")
    parser.add_argument("--split_condition", type=str, default="tv_ratio:4-1", help="Custom string deciding how to split datasets into train/test. Affiliated with a custom function in dataset")
    parser.add_argument("--dataset_path", type=str, nargs="+", default=os.path.expanduser("~/"), help="Dataset paths")

    parser.add_argument_group("Model specific arguments")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "transformer"], help="Type of model to run")
    parser.add_argument("--model_path", type=str, default="", help="path of saved model")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--krnl_size", type=int, default=3, help="Height and width kernel size")
    parser.add_argument("--krnl_size_t", type=int, default=3, help="Temporal kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Height and width Padding")
    parser.add_argument("--padding_t", type=int, default=1, help="Temporal Padding")
    parser.add_argument("--depth", type=int, default=2, help="depth of the updown")
    parser.add_argument("--load_mode", type=str, default="pad", help="in what manner do we create a classification network")
    parser.add_argument("--channel_factor", type=int, default=64, choices=["pad","replace"], help="channel scale factor for up down network")
    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "smooth_l1", "focal", "SSIM"], help="Loss function for the network")
    parser.add_argument("--reduce", action="store_true", help="reduction of loss function toggled")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")

    parser.add_argument_group("Logging arguments")
    parser.add_argument("--visdom", action="store_true", help="use a visdom ploter")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")
    args = parser.parse_args()
    print(args)
    ################################
    # MMNIST
    ################################
    if args.TASK == "MNIST":
        model = FCUp_Down2D_2_MNIST(args, load_path=args.model_path)
        model.to(args.device)
        if args.visdom:
            #args.plotter = VisdomLinePlotter(env_name=args.jobname)
            wandb.init(project="visual-modelling", entity="visual-modelling", name=args.jobname, resume="allow")
            #wandb.config.update(args, allow_val_change=True)
        # Training (includes validation after each epoch and early stopping)
        if args.epoch > 0:        
            best_acc, return_string = train_MNIST(model, args, bsz=args.bsz, epochs=args.epoch)
        # Only Testing
        if args.epoch == 0:
            best_acc, _ = test_MNIST(model, args, bsz=args.val_bsz)
            return_string = f"Validation Only: Accuracy: {best_acc:.2f}%"

    ################################
    # MOCAP
    ################################
    if args.TASK == "MOCAP":
        raise NotImplementedError(f"MOCAP split currently gathered has not proven useful enough to use")
        MOCAP_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/mocap/labels.txt")
        pandas_mocap = pd.read_csv(MOCAP_labels_path, sep="\t")
        label_dict = {}
        for key, row in pandas_mocap.iterrows():
            subj, descr = row["Nested_ID"], row["class/description"]
            if subj[0]=="#":
                descr = descr.replace("(","").replace(")","")
                sub_dict = {"class/description": descr}
                subj_counter = f"{int(subj[1:]):02}"
                print(int(subj[1:]))
                label_dict[subj_counter] = sub_dict
            else:
                label_dict[subj_counter][f"{int(subj):02}"] = descr
        print(label_dict)
        
    ################################
    # HDMB-51
    ################################
    if args.TASK == "HDMB-51":
        #import ipdb; ipdb.set_trace()
        # Labels
        #HDMB_create_labels()
        train_labels, class2id, id2class = utils.load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))
        test_labels, _, _ = utils.load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))

        # Models
        #import ipdb; ipdb.set_trace()
        model = FCUp_Down2D_2_MNIST(args, load_path=args.model_path)
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



    # Print/Logging
    print(return_string) 
    if args.visdom:
        wandb.log({"Run Info" : return_string})
        wandb.log({'Best Accuracy' : best_acc})


