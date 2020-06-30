#!/bin/bash
cd "$(dirname "$0")"
source /home/jumperkables/kable_management/python_venvs/vm/bin/activate
#python ../dataset.py \
#    --raw_frame_rootdir data/hudsons_multi_ygrav/10000 \
#    --extracted_dset_savepathdir data/hudsons_multi_ygrav/10000 \
#    --in_no 5 \
#    --out_no 1 \
#    --extract_dset 
#echo 'Done'
#python ../dataset.py \
#    --raw_frame_rootdir data/hudsons_multi_ygrav/10000_masked \
#    --extracted_dset_savepathdir data/hudsons_multi_ygrav/10000_masked \
#    --in_no 5 \
#    --out_no 1 \
#    --extract_dset 
#echo 'Done'
#python ../dataset.py \
#    --raw_frame_rootdir data/hudsons_multi_ygrav/10000_blurred \
#    --extracted_dset_savepathdir data/hudsons_multi_ygrav/10000_blurred \
#    --in_no 5 \
#    --out_no 1 \
#    --extract_dset 
#echo 'Done'
#python ../dataset.py \
#    --raw_frame_rootdir data/hudsons_multi_ygrav/10000_masked_blurred \
#    --extracted_dset_savepathdir data/hudsons_multi_ygrav/10000_masked_blurred \
#    --in_no 5 \
#    --out_no 1 \
#    --extract_dset 
#echo 'Done'
python ../dataset.py \
    --raw_frame_rootdir data/hudsons_multi_ygrav/2000 \
    --extracted_dset_savepathdir data/hudsons_multi_ygrav/2000 \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 
echo 'Fin'

