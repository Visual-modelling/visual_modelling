#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 5e-5_127-moon_200_moon-regress_5-1_k-3_d-3_ssim  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/5e-5_127-moon_200_moon-regress_5-1_k-3_d-3_ssim.out
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
    --lr 5e-5 \
    --epoch 400 \
    --jobname 5e-5_127-moon_200_moon-regress_5-1_k-3_d-3_ssim \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/moon_5-1_k-3_d-3_lr-1e-3_ssim-epoch=127.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --disable_preload \
    --wandb 
