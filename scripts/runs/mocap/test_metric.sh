#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name toms_metric_test 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/toms_metric_test.out

source ../../../python_venvs/vm/bin/activate
python ../../../main.py \
    --dataset from_raw mmnist \
    --dataset_path data/hudsons_multi_ygrav/500 data/moving_mnist/6_500.npz \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --split_condition tv_ratio:4-1 \
    --device 0 \
    --epoch 300 \
    --early_stopping 100 \
    --jobname toms_metric_test \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --reduce \
    --shuffle \
    --visdom
