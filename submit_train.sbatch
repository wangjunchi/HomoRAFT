#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10240m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --output="hard_lr-4_decay_no_mask_in_model.out"
#SBATCH --open-mode=truncate

python -u train.py