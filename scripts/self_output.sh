#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=MSE_d3_mean
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/cnn_visual_modelling/.results/MSE_d3_mean.out

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

cd /home/jumperkables/cnn_visual_modelling/scripts
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../main.py \
    --dataset_path /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/11_dset.pickle \
    --repo_rootdir /home/jumperkables/cnn_visual_modelling \
    --bsz 1 \
    --in_no 10 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --jobname vm_mk0_MSE_d3_mean_SO \
    --img_type greyscale \
    --self_output \
    --self_output_n 100 \
    --model_path /home/jumperkables/cnn_visual_modelling/.results/vm_mk0_MSE_d3_mean/model.pth \
    --gif_path /home/jumperkables/cnn_visual_modelling/.results/vm_mk0_MSE_d3_mean/d3_MSE_10-1.gif
