#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -t 7-00:00
#SBATCH -p part0
#SBATCH --job-name hdmb51-pt-sl1-mean 
#SBATCH --gres gpu:1
cd ../../../../..
cd $SCRIPT_DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset hdmb51 \
    --dataset_path data/HDMB-51/grey_64x64_frames \
    --split_condition tv_ratio:4-1 \
    --bsz 64 \
    --val_bsz 64 \
    --num_workers 4 \
    --in_no 5 \
    --padding 1 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --jobname hdmb51-pt-sl1-mean \
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
