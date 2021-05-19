#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name no_30_bounces_3d  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/no_30_bounces_3d.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces_3d Task
python test_tasks.py \
    --task bounces \
    --dataset simulations  \
    --dataset_path data/3d_bouncing/3d_regen \
    --bsz 16 \
    --val_bsz 100 \
    --num_workers 0 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 1 \
    --epoch 30 \
    --jobname no_30_bounces_3d \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/mixed_ssim-epoch=72-valid_loss=0.02.ckpt' \
    --shuffle \
    --wandb 
