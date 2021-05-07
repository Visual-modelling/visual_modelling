import os, sys
import argparse
import shutil

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from dataset import Simulations
from models.UpDown2D import FCUp_Down2D
import tools.radam as radam
import tools.loss
import tools.utils as utils

import numpy as np
import tqdm
from tqdm import tqdm
from time import sleep
import pandas as pd
import wandb



################################################################################################
# Pytorch lightning trainers
################################################################################################
class FCUp_Down2D_2_Scalars(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FCUp_Down2D(args)
        
        # Checkpointing
        if args.model_path != "":   # Empty string implies no loading
            checkpoint = torch.load(args.model_path)
            state_dict = checkpoint['state_dict']
            # need to change the names of the state_dict keys from preloaded model
            state_dict = {".".join(mod_name.split(".")[1:]):mod for mod_name, mod in state_dict.items()}
            self.model.load_state_dict(state_dict)

        # Different tasks will be expecting different output numbers
        if args.task == "mnist":
            n_outputs = 10
        else:
            raise NotImplementedError("Task has not been implemented yet")

        # Classifier at the end
        self.cls_mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64*64, 100),
            nn.BatchNorm1d(100),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(100, n_outputs)   # n+1 (includes unknown answer token)
        )

        # Manual optimisation to allow slower training for previous layers
        self.automatic_optimization = False

        # Validation metrics
        self.valid_acc = torchmetrics.Accuracy()

        # Training metrics
        self.train_acc = torchmetrics.Accuracy()
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        model_opt = radam.RAdam([p for p in self.model.parameters() if p.requires_grad], lr=3e-6, weight_decay=1e-5)
        cls_mlp_opt = radam.RAdam([p for p in self.cls_mlp.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
        return model_opt, cls_mlp_opt

    def forward(self, x):
        # Through the encoder
        x = self.model(x)
        # And then the classifier
        out = self.cls_mlp(x)
        return out

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        model_opt, cls_mlp_opt = self.optimizers()
        model_opt.zero_grad()
        cls_mlp_opt.zero_grad()
        frame, label = train_batch
        frame = frame.float()
        frames = frame.repeat(1,self.args.in_no,1,1)
        out = self.model(frames)
        out = out.view(self.args.bsz, -1)
        out = self.cls_mlp(out)
        out = F.softmax(out, dim=1)
        train_loss = self.criterion(out, label)
        self.manual_backward(train_loss)
        model_opt.step()
        cls_mlp_opt.step()
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc(out, label), prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, valid_batch, batch_idx):
        frame, label = valid_batch
        frame = frame.float()
        frames = frame.repeat(1,self.args.in_no,1,1)
        out = self.model(frames)
        out = out.view(self.args.val_bsz, -1)
        out = self.cls_mlp(out)
        out = F.softmax(out, dim=1)
        valid_loss = self.criterion(out, label)
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True)
        self.log("valid_acc", self.valid_acc(out, label), prog_bar=True, on_step=False, on_epoch=True)


if __name__ == "__main__":
    torch.manual_seed(2667)
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Run specific arguments")
    parser.add_argument("--task", type=str, choices=["mnist", "mocap", "hdmb51", "grav", "pendulum"], help="Which task, classification or otherwise, to apply")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0, 1 for appropriate device")
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--val_bsz", type=int, default=100)
    parser.add_argument("--in_no", type=int, default=5, help="number of frames to use for forward pass")
    parser.add_argument("--out_no", type=int, default=1, help="number of frames to use for ground_truth")
    parser.add_argument("--wandb", action="store_true", help="Save models/validation things to checkpoint location")
    parser.add_argument("--jobname", type=str, default="jobname", help="jobname")

    parser.add_argument_group("Dataset specific arguments")
    parser.add_argument("--split_condition", type=str, default="tv_ratio:4-1", help="Custom string deciding how to split datasets into train/test. Affiliated with a custom function in dataset")
    parser.add_argument("--dataset", type=str, nargs="+", choices=["mmnist", "simulations", "mocap", "hdmb51"], help="Type of dataset")
    parser.add_argument("--dataset_path", type=str, nargs="+", default=os.path.expanduser("~/"), help="Dataset paths")
    parser.add_argument("--shuffle", action="store_true", help="shuffle dataset")

    parser.add_argument_group("Model specific arguments")
    parser.add_argument("--model", type=str, default="UpDown2D", choices=["UpDown2D", "UpDown3D", "transformer"], help="Type of model to run")
    parser.add_argument("--model_path", type=str, default="", help="path of saved model")
    parser.add_argument("--img_type", type=str, default="binary", choices=["binary", "greyscale", "RGB"], help="Type of input image")
    parser.add_argument("--krnl_size", type=int, default=3, help="Height and width kernel size")
    parser.add_argument("--krnl_size_t", type=int, default=3, help="Temporal kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Height and width Padding")
    parser.add_argument("--padding_t", type=int, default=1, help="Temporal Padding")
    parser.add_argument("--depth", type=int, default=2, help="depth of the updown")
    parser.add_argument("--channel_factor", type=int, default=64, choices=["pad","replace"], help="channel scale factor for up down network")
    parser.add_argument("--loss", type=str, default="MSE", choices=["mse", "sl1", "focal", "ssim", "mnist"], help="Loss function for the network")
    parser.add_argument("--reduction", type=str, choices=["mean", "sum"], help="type of reduction to apply on loss")

    args = parser.parse_args()
    print(args)

    ########################################################################################################################
    ######## ERROR CONDITIONS To make sure erroneous runs aren't accidentally executed
    if not args.task == "mnist":
        assert len(args.dataset) == len(args.dataset_path), f"Number of specified dataset paths and dataset types should be equal"

    # Mean squared error is naturally only ran with reduction of mean
    if args.loss == "mse":
        assert args.reduction == "mean", f"MSE loss only works with reduction set to mean"
    if (not args.shuffle) and (len(args.dataset_path)>1):
        raise NotImplementedError("Shuffle because multiple self_out data examples from each dataset need to be represented")
    
    # SSIM functional needs to be used. reduction cannot be specified
    if args.loss == "ssim":
        assert args.reduction == "mean", f"SSIM functional needs to be used. cant be bothered to rewrite to allow this for now as its irrelevant. instead default to mean reduction"

    if args.task in ["hdmb51","mocap"]:
        raise NotImplementedError("Haven't reimplemented this yet. May not be worth it in the end.")

    if args.task in ["grav","pendulum"]:
        raise NotImplementedError(f"{args.task} is next to implement")
    ########################################################################################################################
    ########################################################################################################################

    # Create full model path
    if args.model_path != "":
        args.model_path = os.path.join(os.path.dirname(__file__), args.model_path)

    # GPU
    if args.device == -1:
        gpus = None
    else: 
        gpus = [args.device]  # TODO Implement multi GPU support

    # Logging and Saving: If we're saving this run, prepare the neccesary directory for saving things
    wandb_logger = pl.loggers.WandbLogger(project="visual-modelling", name=args.jobname, offline=not args.wandb)#, resume="allow")
    repo_rootdir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_dir = os.path.join(repo_rootdir, ".results", args.jobname )
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)
    args.results_dir = results_dir

    ################################
    # MMNIST
    ################################
    if args.task == "mnist":
        """
        You don't need to set a dataset or dataset path for MNIST. Its all handled here since its so small and easy to load
       """
        train_dset = MNIST(train=True, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
        valid_dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
        train_loader = DataLoader(train_dset, batch_size=args.bsz)
        valid_loader = DataLoader(valid_dset, batch_size=args.val_bsz)
        pl_system = FCUp_Down2D_2_Scalars(args)
               
    ################################
    # HDMB-51
    ################################
    elif args.task == "hdmb51":
        #HDMB_create_labels() # TODO officially allow this function below
        train_labels, class2id, id2class = utils.load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))
        test_labels, _, _ = utils.load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))

        # Models
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

    ################################
    # MOCAP
    ################################
    elif args.task == "mocap":
        raise NotImplementedError(f"MOCAP split currently gathered has not proven useful enough to use")
        #MOCAP_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/mocap/labels.txt")
        #pandas_mocap = pd.read_csv(MOCAP_labels_path, sep="\t")
        #label_dict = {}
        #for key, row in pandas_mocap.iterrows():
        #    subj, descr = row["Nested_ID"], row["class/description"]
        #    if subj[0]=="#":
        #        descr = descr.replace("(","").replace(")","")
        #        sub_dict = {"class/description": descr}
        #        subj_counter = f"{int(subj[1:]):02}"
        #        print(int(subj[1:]))
        #        label_dict[subj_counter] = sub_dict
        #    else:
        #        label_dict[subj_counter][f"{int(subj):02}"] = descr
        #print(label_dict)
 


    # Checkpointing and running
    if args.task in ["mnist", "mocap", "hdmb51"]:   # Accuracy tasks
        max_or_min = "max"
        monitoring = "valid_acc"
    elif args.task in ["grav", "pendulum"]:
        raise NotImplementedError(f"Task: {args.task} has not got its optimisation and task definitions well defined")
        #max_or_min = "min"
        #monitoring = "valid_loss"
    else:
        raise NotImplementedError(f"Task: {arg.task} is not handled")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitoring,
        dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
        filename=f"{args.jobname}"+'-{epoch:02d}-{valid_acc:.2f}',
        save_top_k=1,
        mode=max_or_min,
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=wandb_logger, gpus=gpus)
    trainer.fit(pl_system, train_loader, valid_loader)







################################################################################################
# Deprecated Code, soon to be deleted
################################################################################################
## HDMB
#def test_HDMB_51(model, args, bsz=1):
#    model.eval()
#    args.dset_sze = -1
#    #dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18,0), transforms.ToTensor()]), root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
#    test_loader = DataLoader(dset, batch_size=bsz)
#    correct, total = 0, 0
#    for batch in tqdm(test_loader, total=len(test_loader)):
#        img, label = batch
#        img = (255*img).long().float()
#        #if args.load_mode == "pad":
#        #    new_label = torch.ones(img.shape)
#        #    for x in range(label.shape[0]):
#        #        new_label[x] = new_label[x]*(label[x]*10) # Image of 0s is equal to class 0, image of 10s to class 1, 20s to class 2 etc..
#        #    label = new_label
#        img, label = img.to(args.device), label.to(args.device)
#        out = model(img)
#        #import ipdb; ipdb.set_trace()
#        if args.load_mode == "pad":
#            predicted = torch.mean(out, [1,2,3])    # Average across all outputs and divide by 10 to determine predicted class
#            predicted = torch.round(predicted * 0.1).long()
#            
#        elif args.load_mode == "replace":  
#            _, predicted = torch.max(out.data, 1)
#        total += label.size(0)
#        correct += (predicted == label).sum().item()
#    
#    acc = 100*correct/total 
#    return acc, f"MNIST accuracy: {acc:.2f}%"
#
#
#
#def train_HDMB_51(model, args, epochs=30, bsz=16):
#    """
#    Train on HDMB-51
#    """
#    #import ipdb; ipdb.set_trace()
#    model.train()
#    args.dset_sze = -1
#    dset = Dataset_from_raw(args.dataset_path[0], args)
#    train_loader = DataLoader(dset, batch_size=bsz)
#    best_acc = 0
#    if args.load_mode == "replace":
#        criterion = nn.CrossEntropyLoss()
#    elif args.load_mode == "pad":
#        # Losses
#        if args.loss == "MSE":
#            criterion = torch.nn.MSELoss(reduce=args.reduce, reduction=args.reduction).to(args.device)
#        elif args.loss == "focal":
#            criterion = tools.loss.FocalLoss(reduce=args.reduce, reduction=args.reduction).to(args.device) 
#        elif args.loss == "smooth_l1":
#            criterion = torch.nn.SmoothL1Loss(reduce=args.reduce, reduction=args.reduction).to(args.device)
#        elif args.loss == "SSIM":
#            criterion = tools.loss.ssim(data_range=255, size_average=True, channel=1).to(args.device)
#        else:
#            raise Exception("Loss not implemented")
#
#    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
#    early_stop_count = 0
#    best_epoch = 0
#    for epoch in range(epochs):
#        train_loss = []
#        with tqdm(total=len(train_loader)) as pbar:
#            for batch_idx, batch in enumerate(train_loader):
#                pbar.set_description(f"Training epoch {epoch}/{epochs}")
#                pbar.update(1)
#                #sleep(0.001)
#                img, label = batch
#                #import ipdb; ipdb.set_trace()
#                img = (255*img).long().float()
#                if args.load_mode == "pad":
#                    new_label = torch.ones(img.shape)
#                    for x in range(label.shape[0]):
#                        new_label[x] = new_label[x]*(label[x]*10) # Image of 0s is equal to class 0, image of 10s to class 1, 20s to class 2 etc..
#                    label = new_label
#                img, label = img.to(args.device), label.to(args.device)
#                out = model(img)
#    
#                loss = criterion(out, label)
#                optimizer.zero_grad()
#                loss.backward()
#                optimizer.step()
#                train_loss.append(loss.item())
#            pbar.close()
#        print("Validating...")
#        train_loss = sum(train_loss)/len(train_loss)
#        val_acc, _ = test_HDMB_51(model, args)
#        model.train()
#        if val_acc > best_acc:
#            best_acc = val_acc
#            best_epoch = epoch
#            early_stop_count = 0
#        else:
#            early_stop_count += 1
#            if early_stop_count >= args.early_stopping:
#                return best_acc, f"Early Stopping @ {epoch} epochs. Best accuracy: {best_acc:.3f}% was at epoch {best_epoch}"
#
#        print(f"Epoch:{epoch}/{epochs}, Loss:{train_loss:.5f}, Val Acc:{val_acc:.3f}%, Early Stop:{early_stop_count}/{args.early_stopping}")
#        if args.visdom:
#            wandb.log({'Train Loss' : train_loss})
#            wandb.log({"Valid Accuracy": val_acc})
#    return best_acc, f"{epochs} Epochs (full). Best Accuracy: {best_acc:.3f}% {'was at final epoch' if early_stop_count == 0 else 'was at epoch '+str(best_epoch)}"




################################
# Utility Functions
################################
#def HDMB_create_labels():
#    HDMB51_labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/testTrainMulti_7030_splits")
#    HDMB51_labels = os.listdir(HDMB51_labels_path)
#    train = {}
#    test = {}
#    other = {}
#    for txtfile in HDMB51_labels:
#        fname = txtfile.split(".")[0]
#        ctgry, _ = fname.split("_test_split")
#        #ctrgy, splt = fname.split("_test_split")
#        assert len(fname.split("_test_split")) == 2, f"{txtfile} breaks this formatting"
#        #splt = splt.split(".")[0]
#        
#        labs = open(f"{HDMB51_labels_path}/{txtfile}", "r").readlines()
#        for ele in labs:
#            vid, train_or_test =  ele.split()
#            vid = vid.split(".")[0]                
#            if train_or_test == "0":
#                other[vid] = ctgry
#            elif train_or_test == "1":
#                train[vid] = ctgry
#            elif train_or_test == "2":
#                test[vid] = ctgry
#            else:
#                print(f"Should not have happend: Test-or-test ID:{train_or_test}")
#    sorted_lab_type = sorted(set(train.values()))
#    class2id = { lab:idx for idx, lab in enumerate(sorted_lab_type)}
#    id2class = { idx:lab for idx, lab in enumerate(sorted_lab_type)}
#    train   = ({ vid:class2id[lab] for vid, lab in train.items()}, class2id, id2class)
#    test    = ({ vid:class2id[lab] for vid, lab in test.items()}, class2id, id2class)
#    other   = ({ vid:class2id[lab] for vid, lab in other.items()}, class2id, id2class)
#    utils.save_pickle(train, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/train_labels.pickle"))
#    utils.save_pickle(test , os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/test_labels.pickle"))
#    utils.save_pickle(other, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/HDMB-51/other_labels.pickle"))





