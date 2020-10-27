#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name tomsd3_3d-2d-MMNIST_65536_SSIM-mean 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
#SBATCH -o ../../../../.results/tomsd3_3d-2d-MMNIST_65536_SSIM-mean.out

source ../../../../python_venvs/vm/bin/activate
python ../../../../VM_train.py \
    --dataset_path data/hudsons_multi_ygrav/10000_masked_blurred/6_dset.pickle data/moving_mnist/6_400000.npz data/3d_hudson/6_dset.pickle \
    --hudson_mmnist_mix \
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
    --jobname tomsd3_3d-2d-MMNIST_65536_SSIM-mean \
    --loss SSIM \
    --reduction mean \
    --img_type greyscale \
    --model UpDown2D \
    --self_output \
    --save \
    --shuffle \
    --visdom \
    --reduce \
    --dset_sze 65536 
