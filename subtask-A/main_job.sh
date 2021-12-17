#!/bin/bash
# JOB HEADERS HERE

#SBATCH --partition=gpu
#SBATCH --time=0-01:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=16GB

module load python/3.8.7
pushd /home/yitongli/eecs595/595-final-project/subtask-A/
source venv1/bin/activate

python subtask1.py