#!/bin/bash
#SBATCH --job-name des_train_align         # Job name
#SBATCH --output logs/train/train_%j.log    # Standard output and error log
#SBATCH --partition h100          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=24:0:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:2                  # Request 1 GPU – dette forteller SLURM at du vil bruke GPU-er
#SBATCH --nodelist=hpc11     # Replace with the desired node name

# Load any necessary modules or activate a virtual environment
# module load cuda                      # Example: Load CUDA module

# Commands to run your GPU job
python align_fine_tuning.py --batch_size 32 --accumulation_steps 4 --augmented --epochs 10000 --maximize --lr 0.00000001 --stop_patience 10000 --save --save_every 1000 --verbose --loss_func contrastive --split_name fold_0_aug --validate --ignore_checkpoint --description word --slurm_job_id $SLURM_JOB_ID --slurm_job_desc "Testing augmented data sett on augmented datasett"