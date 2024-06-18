#!/bin/bash
#SBATCH --job-name prep_data         # Job name
#SBATCH --output logs/prep_data_%j.log    # Standard output and error log
#SBATCH --partition normal          # Partition (queue) to submit to
#SBATCH --ntasks=1                   # Number of tasks (processes) to launch
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=4G                     # Memory (default units are megabytes)
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --nodelist=hpc7     # Replace with the desired node name

# Load Python environment or any other required modules
module load python3

# Execute the script
python prepear_data.py -s --folds 0 1 2 3 4 -n aug -a
