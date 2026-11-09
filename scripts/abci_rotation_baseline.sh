#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -J 1-7
#PBS -j oe
#PBS -N rot_baseline

set -euo pipefail
: "${MODEL:?usage: qsub -P <group> -v MODEL=<anysleep|usleep|hypnopc|graphsleepnet|graphsleepnet_distance|mstgcn|productgraphsleepnet>[,CONFIG=native] scripts/abci_rotation_baseline.sh}"
CONFIG="${CONFIG:-native}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=1

group=$((PBS_ARRAY_INDEX - 1))
fold="baselines/folds/rot${group}.json"
mkdir -p logs
log="logs/baseline_${MODEL}_${CONFIG}_rot${group}_${PBS_JOBID%%.*}.log"
{
  echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD), fold ${fold}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} | tee "${log}"
python -m baselines.run "${MODEL}" "${CONFIG}" --fold "${fold}" 2>&1 | tee -a "${log}"
