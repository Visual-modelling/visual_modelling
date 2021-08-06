#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 18G
#SBATCH -p res-gpu-small
#SBATCH --job-name 149-2dB_200_bounces-regress_2d_59-1_k-3_d-3_lr-1e-6_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/149-2dB_200_bounces-regress_2d_59-1_k-3_d-3_lr-1e-6_sl1.out
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
    --lr 1e-6 \
    --epoch 200 \
    --jobname 149-2dB_200_bounces-regress_2d_59-1_k-3_d-3_lr-1e-6_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/2dBouncingMG-y_59-1_k-3_d-3_lr-5e-5_sl1-mean-epoch=149.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 