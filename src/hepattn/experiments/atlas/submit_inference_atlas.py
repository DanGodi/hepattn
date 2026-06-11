"""Submit an inference (test) job to the GPU queue.

Usage:
    python submit_job_nilotpal_inference.py <ckpt_path> [config]

config defaults to configs/base_inference.yaml.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

ckpt = Path(sys.argv[1]).resolve()
config = sys.argv[2] if len(sys.argv) > 2 else "configs/base_inference.yaml"

if not ckpt.is_file():
    sys.exit(f"Checkpoint not found: {ckpt}")

ncpus = "8"
ngpus = "1"
mem = "64gb"
walltime = "08:00:00"

logs_dir = Path.cwd() / "logs"
logs_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

command = f"qsub -o {logs_dir}/pbs_out_infer_{stamp}.log"
command += f" -e {logs_dir}/pbs_err_infer_{stamp}.log"
command += f" -q gpu -N glow_atlas_infer -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},gputype=A6000,io=0.1"
command += f' -v CKPT="{ckpt}",CONFIG="{config}"'
command += f" {Path.cwd()}/run_inference_on_node.sh"

print(command)
os.system(command)
