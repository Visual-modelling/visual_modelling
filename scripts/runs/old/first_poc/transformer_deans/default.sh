#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=deans
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/projects/Visual-modelling/cnn_visual_modelling/.results/deans.out
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../../../../transformer_modelling/train.py \
    --batch_size 16 \
    --lr 0.00001 \
    --epochs 300 \
    --model VMTransformer \
    --config 4096

