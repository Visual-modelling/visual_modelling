#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name random_200_bounces-regress_2d_5-1_k-3_d-3  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/random_200_bounces-regress_2d_5-1_k-3_d-3.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-regress_2d Task
python test_tasks.py \
    --task 2dbounces-regress \
    --dataset simulations  \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/raw \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 59 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-4 \
    --epoch 200 \
    --jobname random_200_bounces-regress_2d_5-1_k-3_d-3 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '' \
    --shuffle \
    --encoder_freeze \
    --linear_probes \
    --wandb 
