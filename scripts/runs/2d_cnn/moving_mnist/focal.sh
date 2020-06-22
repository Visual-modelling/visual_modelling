#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=mmnist_2d-d3_focal-mean
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/mmnist_2d-d3_focal-mean.out

source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../../../../main.py \
    --dataset mmnist \
    --dataset_path data/moving_mnist/mnist_test_seq.npy \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 10 \
    --out_no 10 \
    --depth 3 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 300 \
    --early_stopping 3 \
    --n_gifs 5 \
    --self_output_n 30 \
    --jobname mmnist_2d-d3_focal_mean \
    --loss focal \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce 
