#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name 3dBouncing_transformer_lr2e-5_sl1_2layer_1pixel_pos4_mask_009 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../../../.results/3dBouncing_transformer_lr2e-5_sl1_2layer_1pixel_pos4_mask_009.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python VM_train.py \
    --dataset simulations \
    --dataset_path data/3dBouncing/3dRegen \
    --jobname 3dBouncing_transformer_lr2e-5_sl1_2layer_1pixel_pos4_mask_009 \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 64 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --device 0 \
    --epoch 60 \
    --n_gifs 20 \
    --reduction mean \
    --img_type greyscale \
    --shuffle \
    --wandb \
    --model image_transformer \
    --d_model 4096 \
    --n_layers 3 \
    --nhead 1 \
    --dim_feedforward 4096 \
    --dropout 0.0 \
    --pixel_regression_layers 1 \
    --norm_layer layer_norm \
    --optimiser adam \
    --output_activation hardsigmoid-256 \
    --pos_encoder 4 \
    --mask \
    --loss sl1 \
    --lr 2e-5 \
