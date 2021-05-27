#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name no_50_segmentation_3d_bouncing  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/no_50_segmentation_3d_bouncing.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# segmentation_3d_bouncing Task
python test_tasks.py \
    --task segmentation \
    --dataset simulations \
    --dataset_path data/3dBouncing/3dOld \
    --bsz 16 \
    --val_bsz 100 \
    --num_workers 2 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 50 \
    --jobname no_50_segmentation_3d_bouncing \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '' \
    --shuffle \
    --wandb 
