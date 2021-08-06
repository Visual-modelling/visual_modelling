#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 18G
#SBATCH -p res-gpu-small
#SBATCH --job-name 125-3dB_200_bounces-regress_3d_5-1_k-3_d-3_lr-1e-6_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/125-3dB_200_bounces-regress_3d_5-1_k-3_d-3_lr-1e-6_ssim.out
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
    --lr 1e-6 \
    --epoch 200 \
    --jobname 125-3dB_200_bounces-regress_3d_5-1_k-3_d-3_lr-1e-6_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/3dBouncing_5-1_k-3_d-3_lr-1e-4_ssim-epoch=125.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
