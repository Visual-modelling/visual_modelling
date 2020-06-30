#!/bin/bash
sbatch 5-1_blurred/sl1.sh
sbatch 5-1_blurred/SSIM.sh
sbatch 5-1_masked/sl1.sh
sbatch 5-1_masked/SSIM.sh
sbatch 5-1_masked_blurred/sl1.sh
sbatch 5-1_masked_blurred/SSIM.sh
