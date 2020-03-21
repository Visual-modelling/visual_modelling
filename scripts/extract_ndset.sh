#!/bin/bash
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python /home/jumperkables/cnn_visual_modelling/main.py \
    --raw_frame_rootdir /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0/raw \
    --extract_n_dset_file \
    --extracted_n_dset_savepathdir /home/jumperkables/kable_management/data/visual_modelling/dataset_v1.0 \
    --in_no 5 \
    --out_no 1 