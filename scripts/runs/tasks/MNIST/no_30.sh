#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name no_30_mnist  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/no_30_mnist.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# MNIST Task
python test_tasks.py \
    --task mnist \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 1 \
    --epoch 30 \
    --jobname no_30_mnist \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/mmnist_sl1-epoch=00-valid_loss=120.14.ckpt' \
    --loss mnist \
    --reduction mean \
    --shuffle \
    --wandb 
