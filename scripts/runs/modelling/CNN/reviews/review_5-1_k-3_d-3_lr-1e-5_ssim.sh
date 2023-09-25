#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name review_5-1_k-3_d-3_lr-1e-5_ssim
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/review_5-1_k-3_d-3_lr-1e-5_ssim.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/text_dataset/letters_ds \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 31 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --device 0 \
    --lr 1e-5 \
    --epoch 150 \
    --n_gifs 20 \
    --jobname review_5-1_k-3_d-3_lr-1e-5_ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --disable_preload \
    --wandb
