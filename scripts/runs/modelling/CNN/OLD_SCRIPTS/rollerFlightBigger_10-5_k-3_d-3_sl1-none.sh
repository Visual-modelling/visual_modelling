#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name rollerFlightBigger_10-5_k-3_d-3_sl1-none 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/rollerFlightBigger_10-5_k-3_d-3_sl1-none.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations \
    --dataset_path data/myphysicslab/RollerFlight_10000_bigger \
    --split_condition tv_ratio:4-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 10 \
    --out_no 5 \
    --depth 3 \
    --device 0 \
    --epoch 75 \
    --n_gifs 20 \
    --jobname rollerFlightBigger_10-5_k-3_d-3_sl1-none \
    --loss sl1 \
    --reduction none \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
