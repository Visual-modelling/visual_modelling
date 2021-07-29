#!/bin/bash
#SBATCH --ntasks 6
#SBATCH --mem 16G
#SBATCH -p part0
#SBATCH --job-name attention-pt_TEST  
#SBATCH --gres gpu:2
#SBATCH -o ../../../../.results/attention-pt_TEST.out
cd ../../../..
export PYTHONBREAKPOINT=ipdb.set_trace
source python_venvs/vm/bin/activate

# bounces-regress_2d Task
python pt_attention.py \
    --task bounces-regress \
    --dataset simulations  \
    --dataset_path data/2dBouncing/2dMultiGrav-Y_regen/TEST \
    --bsz 1 \
    --num_workers 1 \
    --in_no 59 \
    --out_no 1 \
    --n_imgs 2 \
    --depth 3 \
    --device 0 \
    --jobname attention-pt_TEST \
    --img_type greyscale \
    --model PatchTrans \
    --model_path '.results/pt_no_200_bounces-regress_2d_5-1_k-3_d-3-epoch=17-valid_loss=4.826.ckpt' \
    --encoder_freeze \
    --linear_probes \
    --shuffle \
    #--wandb 
