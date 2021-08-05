import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from torchvision.datasets import MNIST
import torchmetrics
from argparse import Namespace

from dataset import SimulationsPreloaded
import tools.radam as radam


class FixedOutput(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.constant = nn.Parameter(torch.ones((n_outputs,), dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        batch_size = x.shape[0]
        out = torch.stack(batch_size * [self.constant], dim=0)
        return out


class ImageProbe(nn.Module):
    def __init__(self, in_no, n_outputs):
        super().__init__()
        self.layer = nn.Linear(in_no*64*64, n_outputs)

    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        x = self.layer(x)
        return x


class PLSystem(pl.LightningModule):
    def __init__(self, in_no, n_outputs, mode, lr, task):
        super().__init__()
        self.in_no = in_no
        self.lr = lr
        self.task = task

        if mode == 'fixed_output':
            self.net = FixedOutput(n_outputs)
        elif mode == 'image_probe':
            self.net = ImageProbe(in_no, n_outputs)

        # Validation metrics
        self.valid_acc = torchmetrics.Accuracy()
        # Test metrics
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.net(x)

    def training_step(self, train_batch, batch_idx):
        if self.task == 'mnist':
            frame, label = train_batch
            frames = frame.repeat(1, self.in_no, 1, 1)
        else:
            frames, _, _, label = train_batch
        frames = frames.float()
        out = self.net(frames)

        if self.task == 'mnist':
            loss = F.cross_entropy(out, label)
        else:
            loss = F.smooth_l1_loss(out, label, beta=0.01)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def val_test_step(self, batch, is_valid=True):
        if is_valid:
            name = 'valid'
            acc_function = self.valid_acc
        else:
            name = 'test'
            acc_function = self.test_acc

        if self.task == 'mnist':
            frame, label = batch
            frames = frame.repeat(1, self.in_no, 1, 1)
        else:
            frames, _, _, label = batch
        frames = frames.float()
        out = self.net(frames)

        if self.task == 'mnist':
            out = F.softmax(out)
            acc = acc_function(out, label)
            self.log(f'{name}_acc', acc, on_step=False, on_epoch=True)
        else:
            loss = F.smooth_l1_loss(out, label)
            self.log(f'{name}_loss', loss, on_step=False, on_epoch=True)
            self.log(f'{name}_l1', F.l1_loss(out, label))

    def validation_step(self, valid_batch, batch_idx):
        self.val_test_step(valid_batch, 'valid')

    def test_step(self, test_batch, batch_idx):
        self.val_test_step(test_batch, 'test')

    def configure_optimizers(self):
        optimizer = radam.RAdam([p for p in self.parameters() if p.requires_grad], lr=self.lr)#, weight_decay=1e-5)
        return optimizer


if __name__ == '__main__':
    min_epochs = 1
    max_epochs = 2000
    early_stopping_patience = 50

    batchsize = 64

    lr = 1e-4

    datasets = {
        '2dBouncing': (('2dbounces-regress', 'grav-regress'), 'data/2dBouncing/2dMultiGrav-Y_regen/raw'),
        '3dBouncing': (('3dbounces-regress',), 'data/3dBouncing/3dRegen'),
        'blocks': (('blocks-regress',), 'data/myphysicslab/Blocks_10000'),
        'mmnist': (('mnist',), ''),
        'moon': (('moon-regress',), 'data/myphysicslab/Moon_10000'),
        'pendulum': (('pendulum-regress',), 'data/myphysicslab/Pendulum_10000'),
        'roller': (('roller-regress',), 'data/myphysicslab/RollerFlight_10000_bigger')
    }

    args = Namespace(img_type='greyscale', split_condition='tv_ratio:8-1-1')

    for dataset, (tasks, dataset_path) in datasets.items():
        for task in tasks:
            ################################
            # MMNIST
            ################################
            if task == "mnist":
                """
                You don't need to set a dataset or dataset path for MNIST. Its all handled here since its so small and easy to load
               """
                args.in_no = 5
                args.out_no = 1
                train_dset = MNIST(train=True, transform=transforms.Compose([transforms.Pad(18, 0), transforms.ToTensor()]),
                                   root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
                valid_test_dset = MNIST(train=False, transform=transforms.Compose([transforms.Pad(18, 0), transforms.ToTensor()]),
                                        root=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
                valid_dset = torch.utils.data.Subset(valid_test_dset, list(range(0, len(valid_test_dset) // 2)))
                test_dset = torch.utils.data.Subset(valid_test_dset, list(range(len(valid_test_dset) // 2, len(valid_test_dset))))

            ################################
            # Roller regression/prediction
            ################################   
            elif task in ["roller-regress", "roller-pred"]:
                args.in_no = 5
                args.out_no = 1
                train_dset = SimulationsPreloaded(dataset_path, 'train', 'consecutive', args, yaml_return="roller")
                valid_dset = train_dset.clone('val', 'consecutive')
                test_dset = train_dset.clone('test', 'consecutive')

            ################################
            # Moon regression/prediction
            ################################   
            elif task in ["moon-regress"]:
                args.in_no = 5
                args.out_no = 1
                train_dset = SimulationsPreloaded(dataset_path, 'train', 'consecutive', args, yaml_return="moon")
                valid_dset = train_dset.clone('val', 'consecutive')
                test_dset = train_dset.clone('test', 'consecutive')

            ################################
            # Block mass ratio regression
            ################################   
            elif task in ["blocks-regress"]:
                args.in_no = 49
                args.out_no = 1
                train_dset = SimulationsPreloaded(dataset_path, 'train', 'consecutive', args, yaml_return="blocks")
                valid_dset = train_dset.clone('val', 'consecutive')
                test_dset = train_dset.clone('test', 'consecutive')

            ################################
            # Pendulum
            ################################   
            elif task == "pendulum-regress":
                args.in_no = 5
                args.out_no = 1
                train_dset = SimulationsPreloaded(dataset_path, 'train', 'consecutive', args, yaml_return="pendulum")
                valid_dset = train_dset.clone('val', 'consecutive')
                test_dset = train_dset.clone('test', 'consecutive')

            ################################
            # Gravity regression/prediction
            ################################   
            elif task in ["grav-regress", "grav-pred"]:
                args.in_no = 5
                args.out_no = 1
                train_dset = SimulationsPreloaded(dataset_path, 'train', 'consecutive', args, yaml_return="grav")
                valid_dset = train_dset.clone('val', 'consecutive')
                test_dset = train_dset.clone('test', 'consecutive')

            ################################
            # Ball bounces regression/prediction
            ################################   
            elif task in ["bounces-regress", "bounces-pred"]:
                if dataset == '2dBouncing':
                    args.in_no = 59
                    args.out_no = 1
                else:
                    args.in_no = 99
                    args.out_no = 1
                    
                train_dset = SimulationsPreloaded(dataset_path, 'train', 'consecutive', args, yaml_return="bounces")
                valid_dset = train_dset.clone('val', 'consecutive')
                test_dset = train_dset.clone('test', 'consecutive')

            else:
                raise ValueError(f'Unknown task {task}')

            train_loader = DataLoader(train_dset, batch_size=batchsize, num_workers=1, pin_memory=True, shuffle=True)
            valid_loader = DataLoader(valid_dset, batch_size=batchsize, num_workers=1, pin_memory=True, shuffle=False)
            test_loader = DataLoader(test_dset, batch_size=batchsize, num_workers=1, pin_memory=True, shuffle=False)

            if task == 'mnist':
                n_outputs = 10
                max_or_min = "max"
                monitoring = "valid_acc"
            elif task in ["segmentation", "pendulum-regress", "bounces-regress", "grav-regress", "roller-regress", "moon-regress", "blocks-regress"]:
                n_outputs = 1
                max_or_min = "min"
                monitoring = "valid_loss"
            else:
                raise NotImplementedError(f"Task: {task} is not handled")

            ###############
            # BASELINE
            ###############
            name = f'baseline_{dataset}_{task}'

            wandb.init(entity='visual-modelling', project='visual-modelling', name=name, reinit=True)
            wandb_logger = WandbLogger()

            early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor=monitoring, patience=early_stopping_patience, mode=max_or_min)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor=monitoring,
                dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
                filename=f"{name}" + '-{epoch:02d}-{valid_acc:.2f}',
                save_top_k=1,
                mode=max_or_min,
            )
            callbacks = [checkpoint_callback, early_stopping_callback]

            pl_system = PLSystem(in_no=args.in_no, n_outputs=n_outputs, mode='fixed_output', lr=lr, task=task)

            trainer = pl.Trainer(callbacks=callbacks, logger=wandb_logger, gpus=1, max_epochs=max_epochs, min_epochs=min_epochs)
            trainer.fit(pl_system, train_loader, valid_loader)

            trainer.test(test_dataloaders=test_loader, ckpt_path='best')

            ###############
            # IMAGE PROBE
            ###############
            name = f'image-probe_{dataset}_{task}'

            wandb.init(entity='visual-modelling', project='visual-modelling', name=name, reinit=True)
            wandb_logger = WandbLogger()

            early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(monitor=monitoring, patience=early_stopping_patience, mode=max_or_min)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor=monitoring,
                dirpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), ".results"),
                filename=f"{name}" + '-{epoch:02d}-{valid_acc:.2f}',
                save_top_k=1,
                mode=max_or_min,
            )
            callbacks = [checkpoint_callback, early_stopping_callback]

            pl_system = PLSystem(in_no=args.in_no, n_outputs=n_outputs, mode='image_probe', lr=lr, task=task)

            trainer = pl.Trainer(callbacks=callbacks, logger=wandb_logger, gpus=1, max_epochs=max_epochs, min_epochs=min_epochs)
            trainer.fit(pl_system, train_loader, valid_loader)

            trainer.test(test_dataloaders=test_loader, ckpt_path='best')


