#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
python ../dataset.py \
    --raw_frame_rootdir data/hudsons_multi_xygrav/15000 \
    --extracted_dset_savepathdir data/hudsons_multi_xygrav/15000 \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 

python ../dataset.py \
    --raw_frame_rootdir data/hudsons_multi_xygrav/15000_masked \
    --extracted_dset_savepathdir data/hudsons_multi_xygrav/15000_masked \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 

python ../dataset.py \
    --raw_frame_rootdir data/hudsons_multi_xygrav/15000_blurred \
    --extracted_dset_savepathdir data/hudsons_multi_xygrav/15000_blurred \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 

python ../dataset.py \
    --raw_frame_rootdir data/hudsons_multi_xygrav/15000_masked_blurred \
    --extracted_dset_savepathdir data/hudsons_multi_xygrav/15000_masked_blurred \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 


