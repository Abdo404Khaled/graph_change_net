#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -N baseline

set -euo pipefail
: "${MODEL:?usage: qsub -P <group> -v MODEL=<anysleep|usleep|deepsleepnet|cbramod|hypnopc|graphsleepnet|graphsleepnet_distance|mstgcn|productgraphsleepnet>,CONFIG=<native|ours>[,EXTRA=--quick] scripts/abci_baseline.sh}"
: "${CONFIG:?usage: qsub -P <group> -v MODEL=<model>,CONFIG=<native|ours>[,EXTRA=--quick] scripts/abci_baseline.sh}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=1

mkdir -p logs
log="logs/baseline_${MODEL}_${CONFIG}${EXTRA:+_quick}_${PBS_JOBID%%.*}.log"
{
  echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "cpus $(nproc), memory $(free -g | awk '/Mem/{print $2}') GB"
} | tee "${log}"
python -m baselines.run "${MODEL}" "${CONFIG}" ${EXTRA:-} 2>&1 | tee -a "${log}"
