#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=3d-d2_focal-mean
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/3d-d2_focal-mean.out

source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../../../../main.py \
    --dataset_path data/hudsons_og/2000/6_dset.pickle \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
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
    --jobname poc_5-1_3d-d2_focal_mean \
    --loss focal \
    --reduction mean \
    --img_type greyscale \
    --model UpDown3D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce 
