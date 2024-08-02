#!/bin/bash
#SBATCH --job-name matrix         # Job name
#SBATCH --output logs/matrix/matrix_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=1:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Request 1 GPU – dette forteller SLURM at du vil bruke GPU-er
#SBATCH --nodelist=hpc7     # Replace with the desired node name

# Load any necessary modules or activate a virtual environment
# module load cuda                      # Example: Load CUDA module

# Commands to run your GPU job
python matrix.py --split_name fold_0_aug --nums 1