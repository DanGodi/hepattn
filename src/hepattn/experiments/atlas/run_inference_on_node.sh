#!/bin/bash
source ~/.bashrc
IOTHROTTLE_LIMIT=100
export IOTHROTTLE_LIMIT=100
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common
cd /storage/agrp/dreyet/GLOW/hepattn_nilotpal/src/hepattn/experiments/atlas

mkdir -p logs
JOBID="${PBS_JOBID:-local-$(date +%Y%m%d-%H%M%S)}"
JOBID_SHORT="${JOBID%%.*}"
LOGFILE="logs/infer_${JOBID_SHORT}.log"
echo "Live log: $(pwd)/${LOGFILE}"

CONFIG="${CONFIG:-configs/base_inference.yaml}"
OVERRIDE="${OVERRIDE:-configs/inference_override.yaml}"
: "${CKPT:?CKPT env var must be set to checkpoint path}"

# `-c` is applied left-to-right; OVERRIDE wins (e.g. forces is_inference: true)
python -u main.py test -c "${CONFIG}" -c "${OVERRIDE}" --ckpt_path "${CKPT}" 2>&1 | tee "${LOGFILE}"
