#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name toms_segmentation_test 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/toms_segmentation_test.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..
source python_venvs/vm/bin/activate
python VM_train.py \
    --dataset from_raw \
    --segmentation \
    --dataset_path data/hudson_true_3d_default \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 1 \
    --out_no 1 \
    --depth 3 \
    --split_condition tv_ratio:4-1 \
    --device 0 \
    --epoch 300 \
    --early_stopping 100 \
    --jobname toms_segmentation_test \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --reduce \
    --shuffle \
    --visdom \
    --model_path .results/mmnist_2d-d3_MSE_mean/model.pth
