#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N sgprep

set -euo pipefail
cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p logs
log="logs/sleepgraph_prepare_${PBS_JOBID%%.*}.log"
echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)" | tee "${log}"
python -m baselines.sleepgraph.native 2>&1 | tee -a "${log}"
