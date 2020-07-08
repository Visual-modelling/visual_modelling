#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name multi_gmb_2d-d3_SSIM-mean
#SBATCH --gres gpu:1
#SBATCH -o /home2/crhf63/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/multi_gmb_2d-d3_SSIM-mean.out

source /home2/crhf63/kable_management/python_venvs/vm/bin/activate
python ../../../../../main.py \
    --dataset_path data/hudsons_multi_ygrav/10000_masked_blurred/6_dset.pickle \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 300 \
    --early_stopping 3 \
    --n_gifs 20 \
    --self_output_n 100 \
    --jobname 2poc_5-1_multi_gmb_2d-d3_SSIM_mean \
    --loss SSIM \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce 
