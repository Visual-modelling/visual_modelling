#!/bin/bash
cd /home2/crhf63/kable_management/projects/Visual-modelling/cnn_visual_modelling/scripts/runs/second_poc/2d_cnn/5-1_blurred
sbatch sl1.sh
sbatch SSIM.sh
cd /home2/crhf63/kable_management/projects/Visual-modelling/cnn_visual_modelling/scripts/runs/second_poc/2d_cnn/5-1_masked
sbatch sl1.sh
sbatch SSIM.sh
cd /home2/crhf63/kable_management/projects/Visual-modelling/cnn_visual_modelling/scripts/runs/second_poc/2d_cnn/5-1_masked_blurred
sbatch sl1.sh
sbatch SSIM.sh
