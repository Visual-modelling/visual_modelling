#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name TODO
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/TODO.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python test_tasks.py \
    --task TODO\
    --dataset_path TODO \
    --model_path TODO\
    --linear_probes \
    --encoder_freeze \
    --jobname TODO \
    --dataset simulations \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 64 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --device 0 \
    --epoch 500 \
    --early_stopping 10 \
    --img_type greyscale \
    --shuffle \
    --wandb \
    --model image_transformer \
    --dataset_mode consecutive \
    --d_model 4096 \
    --n_layers 2 \
    --nhead 4 \
    --dim_feedforward 4096 \
    --dropout 0.0 \
    --pixel_regression_layers 1 \
    --norm_layer layer_norm \
    --output_activation hardsigmoid-256 \
    --pos_encoder add \
    --mask \
    --loss sl1 \
    --lr 1e-5 \
