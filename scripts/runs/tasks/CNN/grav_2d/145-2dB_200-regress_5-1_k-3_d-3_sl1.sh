#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name 145-2dB_200_grav_2d-regress_5-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/145-2dB_200_grav_2d-regress_5-1_k-3_d-3_sl1.out
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
    --lr 1e-7 \
    --epoch 400 \
    --jobname 145-2dB_200_grav_2d-regress_5-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/2dBouncingMG-y_5-1_k-3_d-3_lr-1e-3_sl1-mean-epoch=145.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
