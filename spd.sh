#!/bin/bash

#SBATCH -J spd-solar  # name you want to appear on the scima queue
#SBATCH -p sciama3.q  # partition you want job sent to
#SBATCH --nodes=1     # how many nodes you want to use. For the MCMC code only 1 node will work
#SBATCH --cpus-per-task=20  # CPUs to use per node. Put the maximum number depending on what partition you use
#SBATCH --time=6:00:00      # how long you want the job to run for
#SBATCH --array=[0-92]%10   # This relates to the input number of the script. %10 means that a max of 10 individual jobs will run at once
#SBATCH --mem=0             # get max memory from node.
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=             # add your email if you want to be notified about jobs running

cd /mnt/lustre/lhill/SPD/

python main.py ${SLURM_ARRAY_TASK_ID}

