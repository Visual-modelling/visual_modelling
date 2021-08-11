#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 16G
#SBATCH -p res-gpu-small
#SBATCH --job-name 140-roll_200_roller-regress_5-1_k-3_d-3_lr-1e-5_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/140-roll_200_roller-regress_5-1_k-3_d-3_lr-1e-5_ssim.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# roller-regress Task
python test_tasks.py \
    --task roller-regress \
    --dataset simulations  \
    --dataset_path data/myphysicslab/RollerFlight_10000_bigger \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-5 \
    --epoch 400 \
    --jobname 140-roll_200_roller-regress_5-1_k-3_d-3_lr-1e-5_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/rollerFlightBigger_5-1_k-3_d-3_lr-1e-4_ssim-epoch=140.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 