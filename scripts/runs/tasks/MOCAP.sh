#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name toms_mnist_test 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../.results/toms_mnist_test.out
cd ../../..
source python_venvs/vm/bin/activate
python test_tasks.py \
    --TASK MOCAP \
    --bsz 16 \
    --val_bsz 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --epoch 100 \
    --early_stopping 20 \
    --jobname toms_mnist_test \
    --img_type greyscale \
    --model UpDown2D \
    --model_path .results/mmnist_2d-d3_MSE_mean/model.pth \
    --load_mode pad \
    --loss smooth_l1 \
    --reduce \
    --reduction mean 
