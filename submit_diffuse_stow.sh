#!/bin/bash
#SBATCH --job-name=pvdeg-pvrw2025-diffuse-stow         # Job name
#SBATCH --output=dask_job_%j.log    # Standard output (%j for job ID)
#SBATCH --error=dask_job_%j.err     # Standard error (%j for job ID)
#SBATCH --time=02:00:00             # Total run time (hh:mm:ss)
#SBATCH --partition=shared          # Queue/partition to use
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks
#SBATCH --cpus-per-task=2           # CPUs per task
#SBATCH --mem=80G                   # Memory for the job
#SBATCH --account=pvsoiling           # Account name


# i deleted the ntasks here but this might not be required with the slurmrunnner
# define number of tasks
# 2 extra: 1 for scheduler, etc.

module load anaconda3  # kestrel module name
source activate /home/tford/.conda-envs/geospatial

# run file which will save outputs as defined in the script
python /home/tford/dev/notebooks/Riccardio/PV_diffuse_tracker_algorithm/5min_algo.py

conda deactivate
