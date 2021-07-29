#!/bin/bash
#SBATCH --qos short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH --mem 28G
#SBATCH -p res-gpu-small
#SBATCH --job-name baselines
#SBATCH --gres gpu:1 
#SBATCH -o ../../../.results/baselines.out
cd ../../../
source python_venvs/vm/bin/activate
export PYTHONBREAKPOINT=ipdb.set_trace
python baseline_test_tasks.py
