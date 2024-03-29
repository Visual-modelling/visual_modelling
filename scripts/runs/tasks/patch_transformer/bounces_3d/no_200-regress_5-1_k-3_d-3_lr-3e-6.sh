#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pt_no_200_bounces-regress_3d_5-1_k-3_d-3_lr-3e-6  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_no_200_bounces-regress_3d_5-1_k-3_d-3_lr-3e-6.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-regress_3d Task
python test_tasks.py \
    --task 3dbounces-regress \
    --dataset simulations  \
    --dataset_path data/3dBouncing/3dRegen \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 99 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 3e-6 \
    --epoch 400 \
    --jobname pt_no_200_bounces-regress_3d_5-1_k-3_d-3_lr-3e-6 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --shuffle \
    --disable_preload \
    --wandb 
