#!/bin/bash
#SBATCH --qos long-high-prio
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name 30_30_hgrav_sl1 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/30_30_hgrav_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..
source python_venvs/vm/bin/activate

# Pretrain
python VM_train.py \
    --dataset bouncing \
    --dataset_path data/hudson_true_3d_default \
    --split_condition tv_ratio:4-1 \
    --bsz 2 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 1 \
    --epoch 30 \
    --early_stopping 100 \
    --jobname 30_30_hgrav_sl1 \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --reduce \
    --shuffle \
    #--visdom \
    #--save

exit

# Gravity prediction task
python VM_train.py \
    --dataset from_raw \
    --grav_pred \
    --dataset_path data/hudson_true_3d_default \
    --bsz 2 \
    --val_bsz 100 \
    --in_no 5 \
    --model_in_no 5 \
    --out_no 1 \
    --depth 3 \
    --split_condition tv_ratio:4-1 \
    --device 1 \
    --epoch 30 \
    --early_stopping 100 \
    --jobname 30_30_hgrav_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --reduce \
    --shuffle \
    --visdom \
    --save \
    --model_path .results/30_30_hgrav_sl1/model.pth


