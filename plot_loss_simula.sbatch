#!/bin/bash

#SBATCH --job-name="Plot loss"
#SBATCH --partition=defq
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out/std/%j-%x-stdout.txt
#SBATCH --error=out/err/%j-%x-stderr.txt

# Source env
source /global/D1/homes/tobiah/align/bin/activate

python plot_loss.py --nums 1 --save_plot --split_name fold_0_aug