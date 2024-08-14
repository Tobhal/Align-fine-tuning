#!/bin/bash
#SBATCH --job-name lr_1e-4_train_align         # Job name
#SBATCH --output logs/train/train_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                  # Request 1 GPU – dette forteller SLURM at du vil bruke GPU-er
#SBATCH --nodelist=hpc4     # Replace with the desired node name

# Load any necessary modules or activate a virtual environment
# module load cuda                      # Example: Load CUDA module

# Commands to run your GPU job
python align_fine_tuning.py --batch_size 32 --accumulation_steps 4 --epochs 10000 --maximize --lr 0.0001 --stop_patience 500 --save --save_every 1000 --verbose --loss_func contrastive --split_name fold_4_t --validate --slurm_job_id $SLURM_JOB_ID --slurm_job_desc 'Testing learning rate of 1e-4 on normal datasett' --ignore_checkpoint --description phosc_number