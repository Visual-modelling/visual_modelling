#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 28G
#SBATCH -p part0
#SBATCH --job-name 68-mmn_200_mnist_1-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/68-mmn_200_mnist_1-1_k-3_d-3_ssim.out
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
    --jobname 68-mmn_200_mnist_1-1_k-3_d-3_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/mmnist_5-1_k-3_d-3_ssim-epoch=68' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
