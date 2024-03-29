#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pt_pendulumSingleBigger_5-1_k-3_d-3_lr-1e-2_sl1-mean 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_pendulumSingleBigger_5-1_k-3_d-3_lr-1e-2_sl1-mean.out
cd ../../../../..
source python_venvs/vm/bin/activate
export MASTER_PORT=10020
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/myphysicslab/Pendulum_10000 \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --device 0 \
    --lr 1e-2 \
    --epoch 150 \
    --n_gifs 20 \
    --jobname pt_pendulumSingleBigger_5-1_k-3_d-3_lr-1e-2_sl1-mean \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model PatchTrans \
    --shuffle \
    --disable_preload \
    --wandb
