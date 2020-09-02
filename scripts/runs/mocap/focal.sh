#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name tomsd3_3d-2d-mocap_focal-mean 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/tomsd3_3d-2d-mocap_focal-mean.out

source ../../../../python_venvs/vm/bin/activate
python ../../../../main.py \
    --dataset_path data/mocap/grey_64x64_frames/6_dset.pickle
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
    --jobname tomsd3_3d-2d-mocap_focal-mean \
    --loss focal \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce 
