#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 74-mmn_200_mnist_1-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/74-mmn_200_mnist_1-1_k-3_d-3_sl1.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# MNIST Task
python test_tasks.py \
    --task mnist \
    --bsz 16 \
    --val_bsz 100 \
    --num_workers 2 \
    --in_no 1 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 200 \
    --jobname 74-mmn_200_mnist_1-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/mmnist_5-1_k-3_d-3_sl1-mean-epoch=74-valid_loss=7.76.ckpt' \
    --encoder_freeze \
    --shuffle \
    --wandb 
