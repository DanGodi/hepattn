#!/bin/bash
source ~/.bashrc
IOTHROTTLE_LIMIT=100
export IOTHROTTLE_LIMIT=100
# export IOTHROTTLE_VERBOSE=1
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common
cd  /storage/agrp/dreyet/GLOW/hepattn_nilotpal/src/hepattn/experiments/atlas

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Live, per-job log so we don't lose output if PBS overwrites/buffers the -o file.
mkdir -p logs
JOBID="${PBS_JOBID:-local-$(date +%Y%m%d-%H%M%S)}"
JOBID_SHORT="${JOBID%%.*}"
LOGFILE="logs/run_${JOBID_SHORT}.log"
echo "Live log: $(pwd)/${LOGFILE}"

CONFIG="${CONFIG:-configs/base.yaml}"
TRAIN_ARGS=""
if [ -n "${CKPT_PATH}" ]; then
    TRAIN_ARGS="--ckpt_path ${CKPT_PATH}"
fi
python -u main.py fit -c "${CONFIG}" ${TRAIN_ARGS} 2>&1 | tee "${LOGFILE}"
