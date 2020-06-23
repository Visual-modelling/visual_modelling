#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../dataset.py \
    --raw_frame_rootdir data/hudsons_og/2000 \
    --extracted_dset_savepathdir data/hudsons_og/2000 \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 

