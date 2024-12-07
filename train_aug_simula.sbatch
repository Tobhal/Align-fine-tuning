#!/bin/bash

#SBATCH --job-name="train augmented datasett"

##SBATCH --partition=a100q
#SBATCH --partition=hgx2q
##SBATCH --partition=mi210q

#SBATCH --time=14-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --output=out/std/train/%j-%x-stdout.txt
#SBATCH --error=out/err/train/%j-%x-stderr.txt

# Source env
source /global/D1/homes/tobiah/align/bin/activate

python align_fine_tuning.py --batch_size 64 --accumulation_steps 8 --augmented --epochs 10000 --maximize --lr 0.00000001 --stop_patience 10000 --save --save_every 10000 --verbose --loss_func contrastive --split_name fold_0_aug --validate --ignore_checkpoint --description description --slurm_job_id $SLURM_JOB_ID --slurm_job_desc "Testing augmented data sett on augmented datasett"