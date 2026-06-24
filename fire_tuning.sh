#!/bin/bash
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --job-name=rl_swarm_tune
#SBATCH --output=job_outputs/outputs/%A_log.out
#SBATCH --error=job_outputs/errors/%A_error.err
#SBATCH --mail-user=adrian.pietsch@uni-konstanz.de


module load devel/python/3.12.3-gnu-14.2

source ../venv/bin/activate

python -u tune_parameters.py
