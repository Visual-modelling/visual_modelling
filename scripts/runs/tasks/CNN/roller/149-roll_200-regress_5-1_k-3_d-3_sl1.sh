#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 149-roll_200_roller-regress_5-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/149-roll_200_roller-regress_5-1_k-3_d-3_sl1.out
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
    --lr 1e-4 \
    --epoch 200 \
    --jobname 149-roll_200_roller-regress_5-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/rollerFlightBigger_5-1_k-3_d-3_lr-1e-4_sl1-mean-epoch=149.ckpt' \
    --encoder_freeze \
    --shuffle \
    --wandb 