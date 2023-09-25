#!/bin/bash
#SBATCH -p part0
#SBATCH --job-name IT3
#SBATCH --ntasks 6
#SBATCH --gres gpu:1

# This script will run the reviews.py script for all the files in the
# reviews directory.

#echo "CNN"
#cd ./CNN/reviews
#for script in $(find . -name "*.sh"); do
#    sh $script;
#done
echo "image_transformer"
cd ./image_transformer/reviews
for script in $(find . -name "*.sh"); do
    sh $script;
done


# echo "patch_transformer"
# cd ./patch_transformer/reviews
# for script in $(find . -name "*.sh"); do
#     sh $script;
# done

