#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name no_30_segmentation_3d_bouncing  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/no_30_segmentation_3d_bouncing.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# segmentation_3d_bouncing Task
python test_tasks.py \
    --task segmentation \
    --dataset simulations \
    --dataset_path data/3d_bouncing/hudson_true_3d_default \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 1 \
    --epoch 30 \
    --jobname no_30_segmentation_3d_bouncing \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/mixed_ssim-epoch=72-valid_loss=0.02.ckpt' \
    --shuffle \
    --wandb 
