#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 2dBouncingMG-y_5-1_k-5_d-3_ssim 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/2dBouncingMG-y_5-1_k-5_d-3_ssim.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 5 \
    --padding 2 \
    --device 0 \
    --epoch 100 \
    --n_gifs 12 \
    --jobname 2dBouncingMG-y_5-1_k-5_d-3_ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
