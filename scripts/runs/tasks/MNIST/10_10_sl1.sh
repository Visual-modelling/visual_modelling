#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name 10_10_mnist_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/10_10_mnist_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..

source python_venvs/vm/bin/activate

# Pretrain on moving MNIST
python VM_train.py \
    --dataset from_raw \
    --dataset_path data/moving_mnist/various \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --split_condition tv_ratio:4-1 \
    --device 0 \
    --epoch 10 \
    --early_stopping 100 \
    --jobname 10_10_mnist_sl1 \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --reduce \
    --shuffle \
    --visdom \
    --save

# MNIST Task
python test_tasks.py \
    --TASK MNIST \
    --bsz 16 \
    --val_bsz 1 \
    --in_no 1 \
    --model_in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 10 \
    --early_stopping 20 \
    --jobname 10_10_mnist_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path .results/10_10_mnist_sl1/model.pth \
    --load_mode pad \
    --loss smooth_l1 \
    --reduce \
    --reduction mean \
    --visdom \
    --save
