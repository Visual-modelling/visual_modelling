#!/bin/bash
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../visualiser.py \
    --dataset_path /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/6_dset.pickle \
    --visdom \
    --jobname vm_test \
    --checkpoint_path /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/vm_test.pth \
    --device -1 \
    --in_no 5 \
    --out_no 1 \
    --vis_n 1