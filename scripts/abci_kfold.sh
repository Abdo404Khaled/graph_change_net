#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -J 1-5
#PBS -j oe
#PBS -N kfold

set -euo pipefail
: "${EXPERIMENT:?usage: qsub -P <group> -v EXPERIMENT=<plan entry> scripts/abci_kfold.sh}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate

fold=$((PBS_ARRAY_INDEX - 1))
mkdir -p logs
log="logs/${EXPERIMENT}_fold${fold}.log"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "${log}"
python run.py --only "${EXPERIMENT}" \
  --set split.n_folds=5 split.fold="${fold}" loader.num_workers=8 2>&1 | tee -a "${log}"
