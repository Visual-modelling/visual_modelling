#!/bin/bash
"""
Descriptions of each of the arguments. (LEAVE) implies you can leave this arg alone. (CHANGE) implies the opposite.
You of course can find full descriptions in argparse of VM_train, I imagine this is more useful though.

--dataset (LEAVE):
    The kind of dataset to expect when loading. 'from_raw' means it will expect to be pointed to the root directory of videos. 

--dataset_path (CHANGE):
    If --dataset is 'from_raw', point this to the root_dir that contains all the video folders

--bsz:
    Obvious

--val_bsz:
    Obvious

--in_no (CHANGE):
    The number of input images the dataloader should receive. I RECOMMEND USING THIS FOR YOUR TRANSFORMER __init__()

--out_no:
    Similar

--depth (REMOVE):
   Used for my 2D cnn. Feel free to kill, redefine as you will

--split_condition (CHANGE):
    This is how my dataset object decides to split into train and test sets. DEFINE YOUR OWN VERSIONS ANDHANDLETHEM IN dataset.py if you fancy. The default is a train-validation ratio 4-1, 80:20.

--device:
    Obvious

--epoch:
    Obvious

--early_stopping:
    How many epochs without improvement you will tolerate before early stopping.

--jobname (CHANGE):
    The jobname that wandb will use. Best to change

--loss (CHANGE):
    Which loss to use. See VM_train.py's argparse for all details

--reduction (CHANGE):
    Calculating a loss over output images. Do you want to sum the output pixels or average them?

--img_type (LEAVE):
    Read images in greyscale

--model (CHANGE):
    The type of model. Use VM_train.py to handle loading your transformer

--self_output (LEAVE):
    Trigger self_output and metric gathering. After each validation call.

--reduce (LEAVE):
    A boolean on weather or not to do reduction. Handled in tools/loss.py. A BIT REDUNDENT

--shuffle:
    Obvious

--visdom (LEAVE):
    Although its called visdom, this is my handler for plotting to wandb or not. Havent been bothered to change the arg name

--save (LEAVE):
    Handler for saving the model or not.

--model_path:
    If you want to load a pretrained model, supply the path here. REMOVING THIS WILL STOP PRETRAINING
   

@RECOMMENDATIONS:
-You may have to change the batch shape to flatten inputs for transformer.
-You should define all new arguments in a new 'argument group' in argparse for the new transformer.
-You can make similar changes to the MNIST.py file when you are ready to try MNIST classification.
"""

source ../../../python_venvs/vm/bin/activate
python ../../../VM_train.py \
    --dataset from_raw \
    --dataset_path data/hudsons_og/2000 \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --split_condition tv_ratio:4-1 \
    --device 0 \
    --epoch 300 \
    --early_stopping 100 \
    --jobname transformer_startup \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model transformer \
    --self_output \
    --reduce \
    --shuffle \
    --visdom \
    --save \
    --model_path .results/EXAMPLE_MODEL/model.pth \
    \
    --transformer_example_arg perhaps_start_here
