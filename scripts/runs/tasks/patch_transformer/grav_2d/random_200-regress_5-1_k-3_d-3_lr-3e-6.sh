#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pt_random_200_grav_2d-regress_5-1_k-3_d-3_lr-3e-6  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_random_200_grav_2d-regress_5-1_k-3_d-3_lr-3e-6.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# grav_2d-regress Task
python test_tasks.py \
    --task grav-regress \
    --dataset simulations  \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 3e-6 \
    --epoch 200 \
    --jobname pt_random_200_grav_2d-regress_5-1_k-3_d-3_lr-3e-6 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --linear_probes \
    --encoder_freeze \
    --shuffle \
    --disable_preload \
    --wandb 
