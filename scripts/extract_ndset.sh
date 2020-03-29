#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../main.py \
    --raw_frame_rootdir /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/raw \
    --extracted_n_dset_savepathdir /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0 \
    --in_no 10 \
    --out_no 1 \
    --extract_n_dset_file 

