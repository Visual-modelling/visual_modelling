#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_no_200_segmentation_3d_bouncing_1-1_k-3_d-3  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_no_200_segmentation_3d_bouncing_1-1_k-3_d-3.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# segmentation_3d_bouncing Task
python test_tasks.py \
    --task segmentation \
    --dataset simulations \
    --dataset_path data/3dBouncing/3dOld \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 1 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-3 \
    --epoch 400 \
    --jobname pt_no_200_segmentation_3d_bouncing_1-1_k-3_d-3 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --shuffle \
    --wandb 
