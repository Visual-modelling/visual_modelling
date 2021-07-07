#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name 2dBouncing_transformer_lr1e-5_sl1_sequence_1_head_014 
#SBATCH --gres gpu:1 
#SBATCH -o ../../../../../.results/2dBouncing_transformer_lr1e-5_sl1_sequence_1_head_014.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python VM_train.py \
    --dataset simulations \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --jobname 2dBouncing_transformer_lr1e-5_sl1_sequence_1_head_014 \
    --split_condition tv_ratio:8-1-1 \
    --bsz 16 \
    --val_bsz 64 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --reduction mean \
    --img_type greyscale \
    --shuffle \
    --wandb \
    --model image_sequence_transformer \
    --dataset_mode overlap \
    --d_model 4096 \
    --n_layers 2 \
    --nhead 1 \
    --dim_feedforward 4096 \
    --dropout 0.0 \
    --pixel_regression_layers 1 \
    --norm_layer layer_norm \
    --optimiser adam \
    --output_activation hardsigmoid-256 \
    --pos_encoder add_runtime \
    --mask \
    --feedback_training_iters 10 \
    --sequence_loss_factor 0.2 \
    --loss sl1 \
    --lr 1e-5 \
