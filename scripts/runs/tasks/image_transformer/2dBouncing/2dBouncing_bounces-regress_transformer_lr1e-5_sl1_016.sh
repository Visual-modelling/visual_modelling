#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name 2dBouncing_bounces-regress_transformer_lr1e-5_sl1_016.sh
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/2dBouncing_bounces-regress_transformer_lr1e-5_sl1_016.out
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python test_tasks.py \
    --task bounces-regress \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --model_path .results/TODO.ckpt\
    --linear_probes \
    --encoder_freeze \
    --jobname 2dBouncing_bounces-regress_transformer_lr1e-5_sl1_016 \
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
    --d_model 4096 \
    --n_layers 2 \
    --nhead 4 \
    --dim_feedforward 4096 \
    --dropout 0.0 \
    --pixel_regression_layers 1 \
    --norm_layer layer_norm \
    --output_activation hardsigmoid-256 \
    --pos_encoder add_runtime \
    --mask \
    --lr 1e-5 \
