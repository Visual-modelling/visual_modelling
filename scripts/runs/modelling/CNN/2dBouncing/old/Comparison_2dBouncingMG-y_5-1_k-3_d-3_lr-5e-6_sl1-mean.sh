#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 21G
#SBATCH -p res-gpu-small
#SBATCH --job-name Comparison_2dBouncingMG-y_5-1_k-3_d-3_lr-5e-6_sl1-mean 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/Comparison_2dBouncingMG-y_5-1_k-3_d-3_lr-5e-6_sl1-mean.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --lr 1e-5 \
    --val_bsz 100 \
    --num_workers 0 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --lr 5e-6 \
    --device 0 \
    --epoch 125 \
    --n_gifs 12 \
    --jobname Comparison_2dBouncingMG-y_5-1_k-3_d-3_lr-5e-6_sl1-mean \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb