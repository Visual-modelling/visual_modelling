#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 74-2dB_200_bounces-pred_2d_59-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/74-2dB_200_bounces-pred_2d_59-1_k-3_d-3_ssim.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-pred_2d Task
python test_tasks.py \
    --task bounces-pred \
    --dataset simulations  \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --bsz 32 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 59 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 200 \
    --jobname 74-2dB_200_bounces-pred_2d_59-1_k-3_d-3_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/' \
    --encoder_freeze \
    --shuffle \
    --wandb 