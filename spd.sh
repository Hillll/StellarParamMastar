#!/bin/bash

#SBATCH -J spd-solar
#SBATCH -p sciama3.q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH --array=[0-92]%10
#SBATCH --mem=0
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=lewis.hill@port.ac.uk

cd /mnt/lustre/lhill/SPD/

python main.py ${SLURM_ARRAY_TASK_ID}

