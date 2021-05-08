#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name no_30_pendulum  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/no_30_pendulum.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# pendulum Task
python test_tasks.py \
    --task pendulum \
    --dataset simulations  \
    --dataset_path data/visual_modelling/myphysicslab/DEMO_double_pendulum \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 1 \
    --epoch 30 \
    --jobname no_30_pendulum \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/mixed_ssim-epoch=72-valid_loss=0.02.ckpt' \
    --shuffle \
    --wandb 
