#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name 70-roll_200_roller-pred_5-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/70-roll_200_roller-pred_5-1_k-3_d-3_sl1.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# roller-pred Task
python test_tasks.py \
    --task roller-pred \
    --dataset simulations  \
    --dataset_path data/myphysicslab/RollerFlight_10000_bigger \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 6 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 200 \
    --jobname 70-roll_200_roller-pred_5-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model UpDown2D \
    --model_path '.results/rollerFlightBigger_5-1_k-3_d-3_sl1-mean-epoch=70-valid_loss=0.67.ckpt' \
    --encoder_freeze \
    --shuffle \
    --wandb 