#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name pt_random_200_blocks-regress_49-1_k-3_d-3  
#SBATCH --gres gpu:1
#SBATCH -o ../../../../../.results/pt_random_200_blocks-regress_49-1_k-3_d-3.out
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
    --lr 1e-4 \
    --epoch 200 \
    --jobname pt_random_200_blocks-regress_49-1_k-3_d-3 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    --wandb 
