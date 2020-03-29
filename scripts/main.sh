#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../main.py \
    --dataset_path /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/11_dset.pickle \
    --repo_rootdir /home/jumperkables/cnn_visual_modelling \
    --bsz 32 \
    --val_bsz 100 \
    --in_no 10 \
    --out_no 1 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 50 \
    --early_stopping 4 \
    --log_freq 1000 \
    --jobname vm_test \
    --loss smooth_l1 \
    --reduction sum \
    --save \
    --shuffle \
    --visdom \
    --reduce 
