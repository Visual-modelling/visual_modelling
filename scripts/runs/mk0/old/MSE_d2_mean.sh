#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=MSE_d2_mean
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/cnn_visual_modelling/.results/MSE_d2_mean.out

##SBATCH -N 1
##SBATCH -p res-gpu-small
##SBATCH -c 4
##SBATCH -t 2-00:00
##SBATCH -x gpu[0-3]
##SBATCH --qos short
##SBATCH --job-name=MSE_d2_mean
##SBATCH --mem=5G
##SBATCH --gres=gpu:1
##SBATCH -o /home/jumperkables/cnn_visual_modelling/.results/MSE_d2_mean.out

cd /home/jumperkables/cnn_visual_modelling/scripts
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../main.py \
    --dataset_path /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/11_dset.pickle \
    --repo_rootdir /home/jumperkables/cnn_visual_modelling \
    --bsz 32 \
    --val_bsz 100 \
    --in_no 10 \
    --out_no 1 \
    --depth 2 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 30 \
    --early_stopping 3 \
    --log_freq 600 \
    --jobname vm_mk0_MSE_d2_mean \
    --loss MSE \
    --reduction mean \
    --save \
    --shuffle \
    --visdom \
    --reduce 