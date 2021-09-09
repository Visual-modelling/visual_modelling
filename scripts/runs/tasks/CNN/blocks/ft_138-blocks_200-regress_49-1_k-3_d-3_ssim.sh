#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name ft_138-blocks_200_blocks-regress_49-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/ft_138-blocks_200_blocks-regress_49-1_k-3_d-3_ssim.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# blocks-regress Task
python test_tasks.py \
    --task blocks-regress \
    --dataset simulations  \
    --dataset_path data/myphysicslab/Blocks_10000 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 49 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 5e-3 \
    --epoch 400 \
    --jobname ft_138-blocks_200_blocks-regress_49-1_k-3_d-3_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/blocks_49-1_k-3_d-3_lr-1e-3_ssim-epoch=138.ckpt' \
    --shuffle \
    --wandb 
