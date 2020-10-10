#!/bin/bash
#SBATCH -x gpu[0-6]
#SBATCH --mem=12G
#SBATCH --qos short
#SBATCH -t 2-00:00
#SBATCH -N 1
#SBATCH --partition=res-gpu-small
#SBATCH --job-name=3d-d2_sl1-mean
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /home2/crhf63/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/3d-d2_sl1-mean.out

source /home2/crhf63/kable_management/python_venvs/vm/bin/activate
python ../../../../../main.py \
    --dataset_path data/hudsons_og/2000/100_dset.pickle \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 95 \
    --out_no 5 \
    --depth 2 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 300 \
    --early_stopping 3 \
    --n_gifs 5 \
    --self_output_n 30 \
    --channel_factor 32 \
    --krnl_size 3 \
    --krnl_size_t 5 \
    --padding 1 \
    --padding_t 2 \
    --jobname poc_95-5_3d-d2_sl1_mean \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown3D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce 
