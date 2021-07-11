#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 21G
#SBATCH -p part0
#SBATCH --job-name CNNADAM_lr-1e-5_2dBouncingMG-y_5-1_k-3_d-3_lr-1e-4_sl1-mean 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/CNNADAM_lr-1e-5_2dBouncingMG-y_5-1_k-3_d-3_lr-1e-4_sl1-mean.out
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
    --num_workers 0 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --device 1 \
    --lr 1e-4 \
    --epoch 150 \
    --n_gifs 12 \
    --jobname CNNADAM_lr-1e-5_2dBouncingMG-y_5-1_k-3_d-3_lr-1e-4_sl1-mean \
    --loss sl1 \
    --optimiser adam \
    --lr 1e-5 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb