#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N graphchangenet

set -euo pipefail
: "${EXPERIMENT:?usage: qsub -P <group> -v EXPERIMENT=<plan entry>[,EXTRA=--quick] scripts/abci_job.sh}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate

mkdir -p logs
log="logs/${EXPERIMENT}${EXTRA:+_quick}_${PBS_JOBID%%.*}.log"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "${log}"
python run.py --only "${EXPERIMENT}" ${EXTRA:-} 2>&1 | tee -a "${log}"
