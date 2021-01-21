#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name p_hdmb51_pretrain_sl1 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/wntrbtm_hdmb51_pretrain_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../../python_venvs/vm/bin/activate

# Pretrain
python ../../../../VM_train.py \
    --dataset from_raw \
    --dataset_path data/HDMB-51/grey_64x64_frames \
    --split_condition tv_ratio:4-1 \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 300 \
    --early_stopping 100 \
    --jobname wntrbtm_hdmb51_pretrain_sl1 \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --visdom \
    --reduce

# Posttrain
python ../../../../test_tasks.py \
    --TASK HDMB-51 \
    --bsz 16 \
    --split_condition tv_ratio:4-1 \
    --val_bsz 1 \
    --in_no 1 \
    --model_in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 100 \
    --early_stopping 20 \
    --jobname wntrbtm_hdmb51_pretrain_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --dataset from_raw \
    --dataset_path data/HDMB-51/grey_64x64_frames \
    --model_path .results/wntrbtm_hdmb51_pretrain_sl1/model.pth \
    --load_mode pad \
    --loss smooth_l1 \
    --reduce \
    --reduction mean \
    --visdom \
    --save
