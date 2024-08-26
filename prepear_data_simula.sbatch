#!/bin/bash

#SBATCH --job-name="Prepear dataset"
#SBATCH --partition=defq
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out/std/%j-%x-stdout.txt
#SBATCH --error=out/err/%j-%x-stderr.txt

# Source env
source /global/D1/homes/tobiah/align/bin/activate

python prepear_data.py -s --folds 0 1 2 3 4 -n aug -a