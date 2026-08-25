source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common

cd /storage/agrp/dangod/glow_in_atlas/hepattn-nilotpal/src/hepattn/experiments/atlas

export IOTHROTTLE_LIMIT=500
export COMET_API_KEY="AOEbFfElX9nZlXmw5SvKivhFD"
export COMET_WORKSPACE="glow-atlas"
export COMET_PROJECT_NAME="runs"

export PYTHONPATH=/storage/agrp/dangod/glow_in_atlas/hepattn-nilotpal/src:/usr/wipp/conda/24.5.0u/envs/common/lib/python3.11/site-packages

# cwd must be experiments/atlas/ so relative config paths (e.g. configs/atlas_var_transform.yaml) resolve
python -S main.py fit \
    --trainer.devices 1 \
    --trainer.default_root_dir /storage/agrp/dangod/glow_in_atlas/hepattn-nilotpal/experiments \
    --data.num_workers 32
