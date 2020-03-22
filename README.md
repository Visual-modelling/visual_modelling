# CNN Based Visual Modelling
## Features
1) Extraction code for Hudson's dataset v1,
2) Variable in-frame/ground truth frames
3) 2D Fully Convolutional Network Implemented with 5-1 in-out frames with skip connections
4) Fully implemented framework including checkpointing, early stopping, variable loss and visdom logging.
5) Visualisation tool for assessing the outputs of saved models (load_state_dict) side by side with ground truth
## Todo
1) Write first 3D Convolution model
2) Train up 2D version and visualise output of that
3) Update the README with any missing details from extract things needed to run, i.e. flags for running scripts/visdom. **PLEASE TELL ME WHAT I MISS!!**
## Installation
1) Get Hudson's dataset: https://github.com/Visual-modelling/2D-bouncing/releases/tag/v1.0
2) Install requirements.txt:
`pip install -r requirements.txt`
3) Hudson's dataset is a bunch of *videos each with 100 frames and positions at each frame*. **We won't just use 99 frames and then predict the final one**. We will infact split the whole dataset up into **sets of n consecutive frames** (first n-1 for the forward pass, and the final 1 as the ground truth, *no repeats, the spare frames at the end of each video folder are dropped*).
4) We start with **n=6**. **5 frames for forward**, **final 1 as ground truth**. You'll need to extract a '6_dset.pickle' file in a bit.
5) Run `bash scripts/extract_ndset.sh`<br />
**--raw_frame_rootdir** = root directory of Hudson's raw dataset  <br />
**--extracted_n_dset_savepathdir** = root directory of where your ndset will be stored  <br />
**--in_no** = number of frames to be used for forward pass (5)  <br />
**--out_no** = number of frames in ground truth (1)  <br />
**Make sure that in_no+out_no = your desired iteration frame size, 5+1=6**
6) Training and validating handled by. `bash scripts/main.sh`  <br />
**--train_ratio** 0.8 => 80% of dataset for training  <br />
**--dataset_path** = path of extracted n_dset.pickle file  <br />
**Model has not been implemented yet. Put your own model in, write the train/validate functions in main.py and off you go!**
## Visualisation Code
1) Plug in any model you have checkpointed to the visualiser.py file
2) Run `bash scripts/visualiser.sh` specifying:  <br />
**--dataset_path** = path of extracted n_dset.pickle file  <br />
**--checkpoint_path** = path of the checkpointed model file  <br />