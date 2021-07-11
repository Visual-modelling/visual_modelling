#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 21G
#SBATCH -t 7-00:00
#SBATCH -p part0
#SBATCH --job-name mocap_5-1_k-3_d-3_lr-1e-4_sl1-mean 
#SBATCH --gres gpu:1
cd ../../../../..
cd $SCRIPT_DIR/../../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset mocap \
    --dataset_path data/mocap/grey_64x64_frames \
    --split_condition tv_ratio:8-1-1 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 0 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --krnl_size 3 \
    --padding 1 \
    --lr 1e-4 \
    --device 0 \
    --epoch 150 \
    --n_gifs 20 \
    --jobname mocap_5-1_k-3_d-3_lr-1e-4_sl1-mean \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb