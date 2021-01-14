#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name toms_HDMB-51_test 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/toms_HDMB-51_test.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..
source python_venvs/vm/bin/activate
python test_tasks.py \
    --TASK HDMB-51 \
    --bsz 16 \
    --split_condition tv_ratio:4-1 \
    --val_bsz 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 100 \
    --early_stopping 20 \
    --jobname toms_HDMB-51_test \
    --img_type greyscale \
    --model UpDown2D \
    --dataset from_raw \
    --dataset_path data/HDMB-51/grey_64x64_frames \
    --model_path .results/mmnist_2d-d3_MSE_mean/model.pth \
    --load_mode pad \
    --loss smooth_l1 \
    --reduce \
    --reduction mean
