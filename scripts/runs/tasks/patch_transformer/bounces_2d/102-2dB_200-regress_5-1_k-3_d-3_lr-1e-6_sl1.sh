#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 16G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_102-2dB_200_bounces-regress_2d_5-1_k-3_d-3_lr-1e-6_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_102-2dB_200_bounces-regress_2d_5-1_k-3_d-3_lr-1e-6_sl1.out
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
    --epoch 400 \
    --jobname pt_102-2dB_200_bounces-regress_2d_5-1_k-3_d-3_lr-1e-6_sl1 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '.results/pt_2dBouncingMG-y_5-1_k-3_d-3_lr-1e-4_sl1-mean-epoch=102.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 