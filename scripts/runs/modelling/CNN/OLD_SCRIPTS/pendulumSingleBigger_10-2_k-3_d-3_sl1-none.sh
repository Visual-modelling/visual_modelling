#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pendulumSingleBigger_10-2_k-3_d-3_sl1-none 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pendulumSingleBigger_10-2_k-3_d-3_sl1-none.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/myphysicslab/Pendulum_1200_bigger \
    --split_condition tv_ratio:4-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 10 \
    --out_no 2 \
    --depth 3 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --jobname pendulumSingleBigger_10-2_k-3_d-3_sl1-none \
    --loss sl1 \
    --reduction none \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
