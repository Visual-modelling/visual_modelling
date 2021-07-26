#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pt_no_200_bounces-regress_3d_5-1_k-3_d-3  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_no_200_bounces-regress_3d_5-1_k-3_d-3.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-regress_3d Task
python test_tasks.py \
    --task bounces-regress \
    --dataset simulations  \
    --dataset_path data/3dBouncing/3dRegen \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 99 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-4 \
    --epoch 200 \
    --jobname pt_no_200_bounces-regress_3d_5-1_k-3_d-3 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --shuffle \
    --wandb 
