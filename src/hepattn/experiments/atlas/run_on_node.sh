#!/bin/bash
source ~/.bashrc
IOTHROTTLE_LIMIT=100
export IOTHROTTLE_LIMIT=100
# export IOTHROTTLE_VERBOSE=1
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common
cd /srv01/agrp/nilotpal/projects/glow_atlas/hepattn/src/hepattn/experiments/atlas

python main.py fit -c configs/base.yaml
