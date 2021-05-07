#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name np_hdmb51_no_pretrain_sl1 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/wntrbtm_hdmb51_no_pretrain.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ../../../../python_venvs/vm/bin/activate

# Posttrain
python ../../../../test_tasks.py \
    --TASK HDMB-51 \
    --bsz 16 \
    --split_condition tv_ratio:4-1 \
    --val_bsz 1 \
    --in_no 1 \
    --out_no 1 \
    --model_in_no 5 \
    --depth 3 \
    --device 0 \
    --epoch 100 \
    --early_stopping 20 \
    --jobname wntrbtm_hdmb51_no_pretrain_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --dataset_path data/HDMB-51/grey_64x64_frames \
    --load_mode pad \
    --loss smooth_l1 \
    --reduce \
    --reduction mean \
    --visdom \
    --save
