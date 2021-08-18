#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 12G
#SBATCH -p res-gpu-small
#SBATCH --job-name GETTING_RANDOM  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/GETTING_RANDOM.out
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
    --jobname GETTING_RANDOM \
    --lr 1e-4 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '' \
    --linear_probes \
    --encoder_freeze \
    --shuffle \
    #--wandb 
