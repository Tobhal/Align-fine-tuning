#!/bin/bash
#SBATCH --job-name train_align         # Job name
#SBATCH --output logs/train_%j.log    # Standard output and error log
#SBATCH --partition a100          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=72:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Request 1 GPU – dette forteller SLURM at du vil bruke GPU-er
#SBATCH --nodelist=hpc10     # Replace with the desired node name

# Load any necessary modules or activate a virtual environment
# module load cuda                      # Example: Load CUDA module

# Commands to run your GPU job
python align_fine_tuning.py --batch_size 64 --accumulation_steps 4 --augmented --epochs 10000 --maximize --lr 0.00000001 --stop_patience 0 --save --save_every 1000 --verbose --loss_func contrastive --split_name fold_0_aug --validate