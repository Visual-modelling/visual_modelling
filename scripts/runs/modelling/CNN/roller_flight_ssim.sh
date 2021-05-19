#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name roller_flight_ssim 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/roller_flight_ssim.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/myphysicslab/RollerFlight_10000 \
    --split_condition tv_ratio:4-1 \
    --bsz 16 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 1000 \
    --n_gifs 50 \
    --jobname roller_flight_ssim \
    --loss ssim \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
