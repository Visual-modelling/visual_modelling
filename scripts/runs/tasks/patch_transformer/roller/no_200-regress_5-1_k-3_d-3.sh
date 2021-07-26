#!/bin/bash
#SBATCH --ntasks 6
#SBATCH -p part0
#SBATCH --job-name pt_no_200_roller-regress_5-1_k-3_d-3  
#SBATCH --gres gpu:2
#SBATCH -o ../../../../../.results/pt_no_200_roller-regress_5-1_k-3_d-3.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# roller-regress Task
python test_tasks.py \
    --task roller-regress \
    --dataset simulations  \
    --dataset_path data/myphysicslab/RollerFlight_10000_bigger \
    --bsz 64 \
    --val_bsz 100 \
    --num_workers 1 \
    --in_no 5 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-4 \
    --epoch 200 \
    --jobname pt_no_200_roller-regress_5-1_k-3_d-3 \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --shuffle \
    --wandb 
