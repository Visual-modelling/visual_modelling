#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../dataset.py \
    --raw_frame_rootdir data/3d_hudson \
    --extracted_dset_savepathdir data/3d_hudson \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 

