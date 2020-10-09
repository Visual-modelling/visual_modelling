# CNN Based Visual Modelling

## Instructions for Zheming and Hudson
0. `git clone https://github.com/Visual-modelling/cnn_visual_modelling`
1. `cd cnn_visual_modelling && git checkout -b MY_BRANCH` 
2. `pip install -r requirements.txt`
3. `cd scripts/runs/transformer`
4. Edit `--dataset_path` in `transformer.sh` to point to the root directory of all videos of a dataset of your choice
5. Source your virtual environment and run the script. You will reach my debug trace where you will have to handle and change Dean's implementation as you see fit
6. I've left a extended docstring of each arg used in the run script. Please pay attention to the `(CHANGE)` marked ones.
7. `--jobname` will be the name wandb plots it as, please change it appropriately, or remove the `--visdom` flag to prevent logging.
7. I'll add extra instructions for functionality like mixed dataset and running MNIST. For now, please ignore all running scripts outside of `transformer`. Some may not work but I'm keeping them around.
