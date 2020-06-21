#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../visualiser.py \
    --dataset_path data/hudsons_multi/2000/6_dset.pickle \
    --checkpoint_path .results/example_save/vm_test.pth \
    --device -1 \
    --in_no 5 \
    --out_no 1 \
    --vis_n 1 \
    --jobname vm_visualiser_test \
    --visdom

