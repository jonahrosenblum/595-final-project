#!/bin/bash
# JOB HEADERS HERE

#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB
#SBATCH --account=eecs595s001f21_class

# set up job
module load cuda
module load cudnn
module load python/3.8.7
pushd /home/jonaher/595-final-project/subtask-B

source env/bin/activate

# run subtask 2
python3 subtask2.py # -b -m