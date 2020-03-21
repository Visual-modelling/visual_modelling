# CNN Based Visual Modelling
1) Get Hudson's dataset: https://github.com/Visual-modelling/2D-bouncing/releases/tag/v1.0
2) Install requirements.txt:
`pip install -r requirements.txt`
3) Hudson's dataset is a bunch of *videos each with 100 frames and positions at each frame*. **We won't just use 99 frames and then predict the final one**. We will infact split the whole dataset up into **sets of n consecutive frames** (first n-1 for the forward pass, and the final 1 as the ground truth, *no repeats, the spare frames at the end of each video folder are dropped*).
4) We start with *n=6*. **5 frames for forward**, **final 1 as ground truth**. You'll need to extract a '6_dset.pickle' file in a bit.
5) Run `bash scripts/extract_ndset.sh`
-raw_frame_rootdir = root directory of Hudson's raw dataset  
-extracted_n_dset_savepathdir = root directory of where your ndset will be stored  
-in_no = number of frames to be used for forward pass (5)  
-out_no = number of frames in ground truth (1)  
**Make sure that in_no+out_no = your desired iteration frame size, 5+1=6**
6) Run `bash scripts/main.sh`
-train_ratio 0.8 => 80% of dataset for training  
-dataset_path = path of extracted n_dset.pickle file  