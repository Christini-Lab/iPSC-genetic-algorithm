#! /bin/bash -l
 
#SBATCH --partition=panda   # cluster-specific
#SBATCH --nodes=50
#SBATCH --ntasks=50
#SBATCH --job-name=christini-job
#SBATCH --time=08:00:00   # HH/MM/SS
#SBATCH --mem=80G

source ~/.bashrc
spack load -r miniconda3
source activate my_root

echo $SLURM_JOB_NODELIST

python main.py

exit
