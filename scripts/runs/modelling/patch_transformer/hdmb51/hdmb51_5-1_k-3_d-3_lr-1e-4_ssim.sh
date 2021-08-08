#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pt_hdmb51_5-1_k-3_d-3_lr-1e-4_ssim 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_hdmb51_5-1_k-3_d-3_lr-1e-4_ssim.out
cd ../../../../..
source python_venvs/vm/bin/activate
export MASTER_PORT=10012
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
    --lr 1e-4 \
    --device 0 \
    --epoch 150 \
    --n_gifs 20 \
    --jobname pt_hdmb51_5-1_k-3_d-3_lr-1e-4_ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model PatchTrans \
    --shuffle \
    --wandb
