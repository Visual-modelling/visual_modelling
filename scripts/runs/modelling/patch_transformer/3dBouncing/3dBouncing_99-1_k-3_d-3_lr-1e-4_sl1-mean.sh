#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 17G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_3dBouncing_99-1_k-3_d-3_lr-1e-4_sl1-mean 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_3dBouncing_99-1_k-3_d-3_lr-1e-4_sl1-mean.out
cd ../../../../..
source python_venvs/vm/bin/activate
export MASTER_PORT=10006
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/3dBouncing/3dRegen \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 0 \
    --in_no 99 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --device 0 \
    --lr 1e-4 \
    --epoch 150 \
    --n_gifs 20 \
    --jobname pt_3dBouncing_99-1_k-3_d-3_lr-1e-4_sl1-mean \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model PatchTrans \
    --shuffle \
    --wandb
