#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=16:00:00
#PBS -J 1-7
#PBS -j oe
#PBS -N rotation

set -euo pipefail
: "${EXPERIMENT:?usage: qsub -P <group> -v EXPERIMENT=exp15_rotation scripts/abci_rotation.sh}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1

group=$((PBS_ARRAY_INDEX - 1))
mkdir -p logs
log="logs/${EXPERIMENT}_rot${group}.log"
{
  echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} | tee "${log}"
python run.py --only "${EXPERIMENT}" \
  --set split.test_group="${group}" loader.num_workers=8 2>&1 | tee -a "${log}"
