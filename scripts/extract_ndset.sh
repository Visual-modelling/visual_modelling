#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../main.py \
    --raw_frame_rootdir /home/jumperkables/kable_management/data/visual_modelling/hudsons_multi/2000_masked \
    --extracted_n_dset_savepathdir /home/jumperkables/kable_management/data/visual_modelling/hudsons_multi/2000_masked \
    --in_no 5 \
    --out_no 1 \
    --extract_n_dset_file 

