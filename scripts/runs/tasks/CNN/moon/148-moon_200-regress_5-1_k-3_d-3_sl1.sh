#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 21G
#SBATCH -p res-gpu-small
#SBATCH --job-name 148-moon_200_moon-regress_5-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/148-moon_200_moon-regress_5-1_k-3_d-3_sl1.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# moon-regress Task
python test_tasks.py \
    --task moon-regress \
    --dataset simulations  \
    --dataset_path data/myphysicslab/Moon_10000 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-4 \
    --epoch 200 \
    --jobname 148-moon_200_moon-regress_5-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/moon_5-1_k-3_d-3_lr-1e-5_sl1-mean-epoch=148.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
