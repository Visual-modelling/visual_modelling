#!/bin/bash
cd "$(dirname "$0")"
source /home2/crhf63/kable_management/python_venvs/vm/bin/activate
python ../main.py \
    --raw_frame_rootdir data/hudsons_og/2000 \
    --extracted_n_dset_savepathdir data/hudsons_og/2000 \
    --in_no 99 \
    --out_no 1 \
    --extract_n_dset_file 

