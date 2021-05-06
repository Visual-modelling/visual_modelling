#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name mixed_sl1 
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/mixed_sl1.out
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ../../../..
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
# Pretrain
python VM_train.py \
    --dataset simulations simulations simulations simulations simulations simulations \
    --dataset_path data/myphysicslab/DEMO_double_pendulum data/3d_bouncing/hudson_true_3d_default data/2d_bouncing/hudsons_multi_ygrav/10000 data/moving_mnist/1_2_3 data/mocap/grey_64x64_frames data/HDMB-51/grey_64x64_frames \
    --split_condition tv_ratio:4-1 \
    --bsz 16 \
    --val_bsz 100 \
    --num_workers 4 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 1000 \
    --n_gifs 50 \
    --jobname mixed_sl1 \
    --loss sl1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --shuffle \
    --wandb
