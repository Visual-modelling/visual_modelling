#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name mixed-pt-sl1 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/mixed-pt-sl1.out
cd ../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations simulations simulations simulations simulations simulations \
    --dataset_path data/myphysicslab/DEMO_double_pendulum data/3d_bouncing/hudson_true_3d_default data/2d_bouncing/hudsons_multi_ygrav/10000 data/moving_mnist/1_2_3 data/mocap/grey_64x64_frames data/HDMB-51/grey_64x64_frames \
    --split_condition tv_ratio:4-1 \
    --bsz 2 \
    --val_bsz 2 \
    --num_workers 4 \
    --in_no 5 \
    --out_no 1 \
    --device 0 \
    --epoch 1000 \
    --n_gifs 20 \
    --jobname mixed-pt-sl1 \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model pixel_transformer \
    --shuffle \
    --wandb \
    --d_model 4096 \
    --n_layers 6 \
    --nhead 8 \
    --dim_feedforward 16384 \
    --dropout 0.1 \
    --pixel_regression_layer \
    --norm_layer layer_norm
