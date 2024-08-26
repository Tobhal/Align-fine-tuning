#!/bin/bash

#SBATCH --job-name="Train normal datasett"
#SBATCH --partition=dgx2q
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=out/std/%j-%x-stdout.txt
#SBATCH --error=out/err/%j-%x-stderr.txt

# Source env
source /global/D1/homes/tobiah/align/bin/activate

python align_fine_tuning.py --batch_size 32 --accumulation_steps 4 --augmented --epochs 10000 --maximize --lr 0.00000001 --stop_patience 10000 --save --save_every 1000 --verbose --loss_func contrastive --split_name fold_0_aug --validate --ignore_checkpoint --description word --slurm_job_id $SLURM_JOB_ID --slurm_job_desc "Testing augmented data sett on augmented datasett"