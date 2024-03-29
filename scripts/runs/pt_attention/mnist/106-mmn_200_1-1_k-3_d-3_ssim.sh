#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 16G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_106-mmn_200_mnist_1-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_106-mmn_200_mnist_1-1_k-3_d-3_ssim.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# MNIST Task
python test_tasks.py \
    --task mnist \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 1 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 200 \
    --jobname pt_106-mmn_200_mnist_1-1_k-3_d-3_ssim \
    --lr 1e-4 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '.results/pt_mmnist_5-1_k-3_d-3_lr-1e-4_ssim-epoch=106.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
