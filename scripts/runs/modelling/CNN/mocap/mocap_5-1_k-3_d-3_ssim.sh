#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -t 7-00:00
#SBATCH -p part0
#SBATCH --job-name mocap_5-1_k-3_d-3_ssim 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/mocap_5-1_k-3_d-3_ssim.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset mocap \
    --dataset_path data/mocap/grey_64x64_frames \
    --split_condition tv_ratio:4-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 2 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --jobname mocap_5-1_k-3_d-3_ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
