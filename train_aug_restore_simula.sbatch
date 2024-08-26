#!/bin/bash

#SBATCH --job-name="train augmented datasett, restores"
#SBATCH --partition=a100q
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out/std/train/%j-%x-stdout.txt
#SBATCH --error=out/err/train/%j-%x-stderr.txt

# Source env
source /global/D1/homes/tobiah/align/bin/activate

python align_fine_tuning.py --batch_size 80 --accumulation_steps 4 --augmented --epochs 10000 --maximize --lr 0.00000001 --stop_patience 10000 --save --save_every 1000 --verbose --loss_func contrastive --split_name fold_0_aug --validate --description word --slurm_job_id $SLURM_JOB_ID --slurm_job_desc "Testing augmented data sett on augmented datasett, resumed" --checkpoint_path "saved_models/align-fine-tune/fold_0_aug/1"