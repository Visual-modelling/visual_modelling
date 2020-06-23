#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../dataset.py \
    --dest /home/jumperkables/kable_management/data/visual_modelling/moving_mnist/6_400000.npz \
    --dataset mmnist \
    --seq_len 6 \
    --seqs 400000 \
    --num_sz 28 \
    --nums_per_image 2 \
    --extract_dset 

