#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 21G
#SBATCH -t 7-00:00
#SBATCH -p res-gpu-small
#SBATCH --job-name hdmb51_5-1_k-3_d-3_lr-5e-6_ssim 
#SBATCH --gres gpu:1
cd ../../../../..
cd $SCRIPT_DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset hdmb51 \
    --dataset_path data/HDMB-51/grey_64x64_frames \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
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
    --n_gifs 20 \
    --jobname hdmb51_5-1_k-3_d-3_lr-5e-6_ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
