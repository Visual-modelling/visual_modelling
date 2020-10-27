#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name varysize-16384_3dhudson_2d-d3_sl1-mean
#SBATCH --gres gpu:1
#SBATCH -o /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/varysize-16384_3dhudson_2d-d3_sl1-mean.out

source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../../../../../../VM_train.py \
    --dataset_path data/3d_hudson/6_dset.pickle \
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
    --jobname jmprkbls_varysize-16384_5-1_3dhudson_2d-d3_sl1_mean \
    --loss smooth_l1 \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce \
    --dset_sze 16384 
