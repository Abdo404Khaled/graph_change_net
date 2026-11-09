#!/bin/bash
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N ablprep

set -euo pipefail
: "${ABLATION:?usage: qsub -P <group> -v ABLATION=<name> scripts/abci_ablation_prepare.sh}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1

mkdir -p logs
log="logs/ablation_prepare_${ABLATION}_${PBS_JOBID%%.*}.log"
echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)" | tee "${log}"
python run.py --config "ablations/configs/${ABLATION}.yaml" --set split.fold=0 --prepare-data 2>&1 | tee -a "${log}"
