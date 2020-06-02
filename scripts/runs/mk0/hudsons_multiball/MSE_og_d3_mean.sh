#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=mk0_hudsonmulti_MSE_d3_mean
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/mk0_hudsonmulti_MSE_d3_mean.out

##SBATCH -N 1
##SBATCH -p res-gpu-small
##SBATCH -c 4
##SBATCH -t 2-00:00
##SBATCH -x gpu[0-3]
##SBATCH --qos short
##SBATCH --job-name=MSE_d3_mean
##SBATCH --mem=5G
##SBATCH --gres=gpu:1
##SBATCH -o /home/jumperkables/cnn_visual_modelling/.results/MSE_d3_mean.out

source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../../../../main.py \
    --dataset_path /home/jumperkables/kable_management/data/visual_modelling/hudsons_multi/2000/6_dset.pickle \
    --repo_rootdir /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling \
    --bsz 8 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 100 \
    --early_stopping 3 \
    --n_gifs 3 \
    --self_output_n 30 \
    --jobname vm_mk0_hudsonmulti_MSE_d3_mean \
    --loss MSE \
    --reduction mean \
    --img_type greyscale \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce 
