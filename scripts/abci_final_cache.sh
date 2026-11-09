#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N final_cache

set -euo pipefail
: "${EXPERIMENT:=final12_window_graph}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=1

mkdir -p logs
log="logs/final_cache_${PBS_JOBID%%.*}.log"
echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)" | tee "${log}"
python run.py --only "${EXPERIMENT}" --prepare-data 2>&1 | tee -a "${log}"
