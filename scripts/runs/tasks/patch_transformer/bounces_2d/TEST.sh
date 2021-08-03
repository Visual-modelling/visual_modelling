#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 16G
#SBATCH -p part0
#SBATCH --job-name TEST_pt_tasks  
#SBATCH --gres gpu:2
#SBATCH -o ../../../../../.results/TEST_pt_tasks.out
cd ../../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-regress_2d Task
python test_tasks.py \
    --task 2dbounces-regress \
    --dataset simulations  \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/TEST \
    --bsz 2 \
    --val_bsz 2 \
    --num_workers 1 \
    --in_no 59 \
    --out_no 1 \
    --depth 3 \
    --device 0 \
    --lr 1e-4 \
    --epoch 200 \
    --jobname TEST_pt_tasks \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    #--wandb 
