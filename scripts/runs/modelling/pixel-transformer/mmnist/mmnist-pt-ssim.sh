#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name mmnist-pt-ssim 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/mmnist-pt-ssim.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/moving_mnist/1_2_3 \
    --split_condition tv_ratio:4-1 \
    --bsz 2 \
    --val_bsz 2 \
    --num_workers 4 \
    --in_no 5 \
    --out_no 1 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --jobname mmnist-pt-ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model pixel_transformer \
    --shuffle \
    --wandb \
    --d_model 4096 \
    --n_layers 6 \
    --nhead 8 \
    --dim_feedforward 16384 \
    --dropout 0.1 \
    --pixel_regression_layer \
    --norm_layer layer_norm
