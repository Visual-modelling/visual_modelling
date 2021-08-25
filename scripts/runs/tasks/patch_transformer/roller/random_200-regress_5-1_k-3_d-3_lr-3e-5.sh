#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 22G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_random_200_roller-regress_5-1_k-3_d-3_lr-3e-5  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_random_200_roller-regress_5-1_k-3_d-3_lr-3e-5.out
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
    --lr 3e-5 \
    --epoch 200 \
    --jobname pt_random_200_roller-regress_5-1_k-3_d-3_lr-3e-5 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
