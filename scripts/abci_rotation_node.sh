#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -N rotation_node

set -euo pipefail
EXPERIMENT="${EXPERIMENT:-exp15_rotation}"
ABLATION="${ABLATION:-unsigned_graph}"

cd "${PBS_O_WORKDIR}"
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=1

mkdir -p logs
{
  echo "job ${PBS_JOBID} on $(hostname), $(date), commit $(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  echo "cpus $(nproc), memory $(free -g | awk '/Mem/{print $2}') GB"
} | tee "logs/rotation_node_${PBS_JOBID%%.*}.log"

pids=()
for group in 0 1 2 3 4 5 6; do
  log="logs/${EXPERIMENT}_rot${group}.log"
  echo "job ${PBS_JOBID} gpu ${group}, $(date), commit $(git rev-parse --short HEAD)" > "${log}"
  CUDA_VISIBLE_DEVICES="${group}" python run.py --only "${EXPERIMENT}" \
    --set split.test_group="${group}" loader.num_workers=8 >> "${log}" 2>&1 &
  pids+=($!)
done
if [ -n "${ABLATION}" ]; then
  log="logs/ablation_${ABLATION}_fold0.log"
  echo "job ${PBS_JOBID} gpu 7, $(date), commit $(git rev-parse --short HEAD)" > "${log}"
  CUDA_VISIBLE_DEVICES=7 python -m ablations.run "${ABLATION}" --fold 0 >> "${log}" 2>&1 &
  pids+=($!)
fi

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
