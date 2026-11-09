#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=36:00:00
#PBS -J 1-5
#PBS -j oe
#PBS -N final

set -euo pipefail
: "${EXPERIMENT:?usage: qsub -P <group> -v EXPERIMENT=<plan entry> scripts/abci_final.sh}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=1

fold=$((PBS_ARRAY_INDEX - 1))
mkdir -p logs
log="logs/${EXPERIMENT}_fold${fold}.log"
{
  echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "cpus $(nproc), memory $(free -g | awk '/Mem/{print $2}') GB, /dev/shm $(df -h /dev/shm | awk 'NR==2{print $2}')"
} | tee "${log}"
python run.py --only "${EXPERIMENT}" --set split.fold="${fold}" 2>&1 | tee -a "${log}"
