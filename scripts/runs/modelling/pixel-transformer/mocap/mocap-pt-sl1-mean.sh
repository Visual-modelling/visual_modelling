#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH -p res-gpu-small
#SBATCH --job-name mocap-pt-sl1-mean 
#SBATCH --gres gpu:1
cd ../../../../..
cd $SCRIPT_DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset mocap \
    --dataset_path data/mocap/grey_64x64_frames \
    --split_condition tv_ratio:4-1 \
    --bsz 64 \
    --val_bsz 64 \
    --num_workers 4 \
    --in_no 5 \
    --padding 1 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --jobname mocap-pt-sl1-mean \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model pixel_transformer \
    --shuffle \
    --d_model 4096 \
    --n_layers 6 \
    --nhead 8 \
    --dim_feedforward 16384 \
    --dropout 0.1 \
    --pixel_regression_layer \
    --norm_layer layer_norm
    --wandb \
