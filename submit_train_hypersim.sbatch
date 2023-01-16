#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10240m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1024

python -u train_hypersim.py