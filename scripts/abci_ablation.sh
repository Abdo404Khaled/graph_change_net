#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -N ablation

set -euo pipefail
: "${ABLATION:?usage: qsub -P <group> -v ABLATION=<name>,FOLD=<k> scripts/abci_ablation.sh, or -J 1-5 for all folds}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=1

fold="${FOLD:-$((PBS_ARRAY_INDEX - 1))}"
mkdir -p logs
log="logs/ablation_${ABLATION}_fold${fold}.log"
{
  echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "cpus $(nproc), memory $(free -g | awk '/Mem/{print $2}') GB, /dev/shm $(df -h /dev/shm | awk 'NR==2{print $2}')"
} | tee "${log}"
python -m ablations.run "${ABLATION}" --fold "${fold}" 2>&1 | tee -a "${log}"
