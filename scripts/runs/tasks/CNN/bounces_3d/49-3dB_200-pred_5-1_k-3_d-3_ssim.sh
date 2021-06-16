#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 49-3dB_200_bounces-pred_3d_5-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/49-3dB_200_bounces-pred_3d_5-1_k-3_d-3_ssim.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-pred_3d Task
python test_tasks.py \
    --task bounces-pred \
    --dataset simulations  \
    --dataset_path data/3dBouncing/3dRegen \
    --bsz 32 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 59 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 200 \
    --jobname 49-3dB_200_bounces-pred_3d_5-1_k-3_d-3_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/3dBouncing_5-1_k-3_d-3_ssim-epoch=49-valid_loss=0.01.ckpt' \
    --encoder_freeze \
    --shuffle \
    --wandb 
