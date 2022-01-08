#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name IT2 
#SBATCH --ntasks 6
#SBATCH --gres gpu:1
########
# CNN
########

## 2d Bouncing
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/2dBouncing
#bash 2dBouncingMG-y_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash 2dBouncingMG-y_5-1_k-3_d-3_lr-1e-4_ssim.sh
#bash 2dBouncingMG-y_59-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash 2dBouncingMG-y_59-1_k-3_d-3_lr-1e-3_ssim.sh
#
## 3d Bouncing 
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/3dBouncing
#bash 3dBouncing_5-1_k-3_d-3_lr-1e-4_sl1-mean.sh
#bash 3dBouncing_5-1_k-3_d-3_lr-1e-3_ssim.sh
#bash 3dBouncing_99-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash 3dBouncing_99-1_k-3_d-3_lr-1e-3_ssim.sh
#
## Blocks
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/blocks
#bash blocks_49-1_k-3_d-3_lr-1e-3_sl1-mean.sh 
#bash blocks_49-1_k-3_d-3_lr-1e-3_ssim.sh
#
## HMDB-51
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/hdmb51
#bash hdmb51_5-1_k-3_d-3_lr-1e-4_sl1-mean.sh
#bash hdmb51_5-1_k-3_d-3_lr-1e-4_ssim.sh
#
## MMNIST
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/mmnist
#bash mmnist_5-1_k-3_d-3_lr-1e-2_sl1-mean.sh
#bash mmnist_5-1_k-3_d-3_lr-1e-2_ssim.sh
#
## MOCAP 
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/mocap
#bash mocap_5-1_k-3_d-3_lr-1e-4_sl1-mean.sh
#bash mocap_5-1_k-3_d-3_lr-1e-4_ssim.sh
#
## Moon
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/moon
#bash moon_5-1_k-3_d-3_lr-1e-2_sl1-mean.sh
#bash moon_5-1_k-3_d-3_lr-1e-3_ssim.sh
#
## Pendulum
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/pendulum
#bash pendulumSingleBigger_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash pendulumSingleBigger_5-1_k-3_d-3_lr-1e-4_ssim.sh
#
## Roller
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/CNN/roller
#bash rollerFlightBigger_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash rollerFlightBigger_5-1_k-3_d-3_lr-1e-3_ssim.sh
#

########
# Image Transformer
#########
# 2d Bouncing
cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/image_transformer/test_only
#bash 2dBouncing_transformer_lr1e-5_sl1_021.sh
#bash 2dBouncing_transformer_lr1e-5_sl1_59in_021.sh
#bash 2dBouncing_transformer_lr1e-5_ssim_021.sh
#bash 2dBouncing_transformer_lr3e-6_ssim_59in_021.sh
#
## 3d Bouncing
#bash 3dBouncing_transformer_lr3e-6_sl1_021.sh
#bash 3dBouncing_transformer_lr3e-6_sl1_99in_021.sh
#bash 3dBouncing_transformer_lr1e-5_ssim_021.sh
#bash 3dBouncing_transformer_lr1e-5_ssim_99in_021.sh
#
## Blocks
#bash blocks_transformer_lr3e-6_sl1_021.sh
#bash blocks_transformer_lr3e-6_ssim_021.sh
#
## HDMB-51 
#bash hdmb51_transformer_lr3e-6_sl1_021.sh
#bash hdmb51_transformer_lr3e-6_ssim_021.sh

# MMNIST
bash mmnist_transformer_lr3e-6_sl1_021.sh
bash mmnist_transformer_lr3e-6_ssim_021.sh

# MOCAP
bash mocap_transformer_lr1e-5_sl1_021.sh
bash mocap_transformer_lr1e-5_ssim_021.sh

# Moon
bash moon_transformer_lr1e-5_sl1_021.sh
bash moon_transformer_lr3e-6_ssim_021.sh

# Pendulum
bash pendulum_transformer_lr3e-6_sl1_021.sh
bash pendulum_transformer_lr3e-6_ssim_021.sh

# Roller
bash roller_transformer_lr1e-5_sl1_021.sh
bash roller_transformer_lr3e-6_ssim_021.sh


#########
## Patch Transformer
#########
#
## 2d Bouncing
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/2dBouncing
#bash 2dBouncingMG-y_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash 2dBouncingMG-y_5-1_k-3_d-3_lr-1e-3_ssim.sh
#bash 2dBouncingMG-y_59-1_k-3_d-3_lr-1e-4_sl1-mean.sh
#bash 2dBouncingMG-y_59-1_k-3_d-3_lr-1e-4_ssim.sh
#
## 3d Bouncing
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/3dBouncing
#bash 3dBouncing_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash 3dBouncing_5-1_k-3_d-3_lr-1e-3_ssim.sh
#bash 3dBouncing_99-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash 3dBouncing_99-1_k-3_d-3_lr-1e-3_ssim.sh
#
## Blocks
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/blocks
#bash blocks_49-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash blocks_49-1_k-3_d-3_lr-1e-3_ssim.sh
#
## HDMB-51 
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/hdmb51
#bash hdmb51_5-1_k-3_d-3_lr-1e-4_sl1-mean.sh
#bash hdmb51_5-1_k-3_d-3_lr-1e-4_ssim.sh 
#
## MMNIST
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/mmnist
#bash mmnist_5-1_k-3_d-3_lr-5e-4_sl1-mean.sh
#bash mmnist_5-1_k-3_d-3_lr-5e-4_ssim.sh
#
## MOCAP
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/mocap
#bash mocap_5-1_k-3_d-3_lr-1e-4_sl1-mean.sh
#bash mocap_5-1_k-3_d-3_lr-1e-4_ssim.sh
#
## Moon
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/moon
#bash moon_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash moon_5-1_k-3_d-3_lr-1e-3_ssim.sh
#
## Pendulum
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/pendulum
#bash pendulumSingleBigger_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash pendulumSingleBigger_5-1_k-3_d-3_lr-1e-4_ssim.sh 
#
## Roller
#cd /home/jumperkables/kable_management/projects/Visual-modelling/visual_modelling/scripts/runs/modelling/patch_transformer/roller
#bash rollerFlightBigger_5-1_k-3_d-3_lr-1e-3_sl1-mean.sh
#bash rollerFlightBigger_5-1_k-3_d-3_lr-1e-3_ssim.sh 
