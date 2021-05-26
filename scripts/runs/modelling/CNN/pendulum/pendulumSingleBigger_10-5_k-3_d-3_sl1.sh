#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name pendulumSingleBigger_10-5_k-3_d-3_sl1 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pendulumSingleBigger_10-5_k-3_d-3_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/myphysicslab/Pendulum_1200_bigger \
    --split_condition tv_ratio:4-1 \
    --bsz 16 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 10 \
    --out_no 5 \
    --depth 3 \
    --device 0 \
    --epoch 50 \
    --n_gifs 20 \
    --jobname pendulumSingleBigger_10-5_k-3_d-3_sl1 \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
