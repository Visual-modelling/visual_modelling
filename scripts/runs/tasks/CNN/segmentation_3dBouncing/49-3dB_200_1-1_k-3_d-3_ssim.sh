#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name 5e-5_49-3dB_200_segmentation_3d_bouncing_1-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/5e-5_49-3dB_200_segmentation_3d_bouncing_1-1_k-3_d-3_ssim.out
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
    --lr 1e-4 \
    --epoch 200 \
    --jobname 5e-5_49-3dB_200_segmentation_3d_bouncing_1-1_k-3_d-3_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/3dBouncing_5-1_k-3_d-3_lr-1e-4_ssim-epoch=49' \
    --shuffle \
    --wandb 
