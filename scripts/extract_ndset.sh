#!/bin/bash
source /home/user/python_venvs/vm/bin/activate
python ../cnn_visual_modelling/main.py \
    --raw_frame_rootdir /home/user/data/visual_modelling/dataset_v1.0/raw \
    --extract_n_dset_file \
    --extracted_n_dset_savepathdir /home/user/data/visual_modelling/dataset_v1.0 \
    --in_no 5 \
    --out_no 1 