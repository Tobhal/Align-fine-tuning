#!/bin/bash
#SBATCH --job-name plot_loss            # Job name
#SBATCH --output logs/plot/plot_loss_%j.log  # Standard output and error log
#SBATCH --partition normal              # Partition (queue) to submit to
#SBATCH --ntasks=1                      # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=4G                        # Memory (default units are megabytes)
#SBATCH --time=1:00:00                 # Time limit hrs:min:sec

python plot_loss.py --nums 6 --save_plot --split_name fold_0_aug