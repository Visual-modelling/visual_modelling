#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 68-2dB_200_grav_2d-pred_5-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/68-2dB_200_grav_2d-pred_5-1_k-3_d-3_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# grav_2d-pred Task
python test_tasks.py \
    --task grav-pred \
    --dataset simulations  \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --bsz 32 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 200 \
    --jobname 68-2dB_200_grav_2d-pred_5-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/2dBouncingMG-y_5-1_k-3_d-3_sl1-mean-epoch=68-valid_loss=0.96.ckpt' \
    --encoder_freeze \
    --shuffle \
    --wandb 