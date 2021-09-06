#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_ft_142-blocks_200_blocks-regress_49-1_k-3_d-3_sl1  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_ft_142-blocks_200_blocks-regress_49-1_k-3_d-3_sl1.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# blocks-regress Task
python test_tasks.py \
    --task blocks-regress \
    --dataset simulations  \
    --dataset_path data/myphysicslab/Blocks_10000 \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 49 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-3 \
    --epoch 400 \
    --jobname pt_ft_142-blocks_200_blocks-regress_49-1_k-3_d-3_sl1 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '.results/pt_blocks_49-1_k-3_d-3_lr-1e-3_sl1-mean-epoch=142.ckpt' \
    --shuffle \
    --wandb 
