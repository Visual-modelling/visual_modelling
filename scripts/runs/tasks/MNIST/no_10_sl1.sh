#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name no_10_mnist_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/no_10_mnist_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..

source python_venvs/vm/bin/activate

# MNIST Task
python test_tasks.py \
    --TASK MNIST \
    --bsz 16 \
    --val_bsz 1 \
    --dataset_path data/moving_mnist/various \
    --in_no 1 \
    --model_in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 10 \
    --early_stopping 20 \
    --jobname no_10_mnist_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --load_mode pad \
    --loss smooth_l1 \
    --reduce \
    --reduction mean \
    --visdom \
    --save
