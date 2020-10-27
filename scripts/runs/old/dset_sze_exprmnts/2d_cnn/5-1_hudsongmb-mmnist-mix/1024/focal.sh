#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name varysize-1024_mixed_2d-d3_focal-mean
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/varysize-1024_mixed_2d-d3_focal-mean.out

source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../../../../../../VM_train.py \
    --dataset_path data/hudsons_multi_ygrav/10000_masked_blurred/6_dset.pickle \
    --hudson_mmnist_mix \
    --MMNIST_path /home/jumperkables/kable_management/data/visual_modelling/moving_mnist/6_400000.npz \
    --bsz 16 \
    --val_bsz 100 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --train_ratio 0.8 \
    --device 0 \
    --epoch 300 \
    --early_stopping 100 \
    --n_gifs 16 \
    --self_output_n 100 \
    --jobname jmprkbls_varysize-1024_5-1_mixed_2d-d3_focal_mean \
    --loss focal \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce \
    --dset_sze 1024 
