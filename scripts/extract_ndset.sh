#!/bin/bash
cd "$(dirname "$0")"
source ../python_venvs/vm/bin/activate
python ../dataset.py \
    --raw_frame_rootdir data/mocap/grey_128x80_frames \
    --extracted_dset_savepathdir data/mocap/grey_128x80_frames \
    --in_no 5 \
    --out_no 1 \
    --extract_dset 

