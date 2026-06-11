import argparse
import os
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(description="Submit training job")
parser.add_argument("--ckpt_path", default=None, help="Path to checkpoint to resume from")
parser.add_argument("--ncpus", default="48", help="Number of CPUs")
parser.add_argument("--ngpus", default="4", help="Number of GPUs")
parser.add_argument("--mem", default="200gb", help="Memory")
parser.add_argument("--walltime", default="72:00:00", help="Wall time")
args = parser.parse_args()

ncpus = args.ncpus
ngpus = args.ngpus
mem = args.mem
walltime = args.walltime

logs_dir = Path.cwd() / "logs"
logs_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Build environment variables for checkpoint
env_vars = ""
if args.ckpt_path:
    env_vars = f" CKPT_PATH={args.ckpt_path}"

command = f"qsub -o {logs_dir}/pbs_out_{stamp}.log"
command += f" -e {logs_dir}/pbs_err_{stamp}.log"
command += f" -q gpu -N glow_atlas -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=0.1,gputype=A6000"
command += f" -v{env_vars}" if env_vars else ""
command += f" {Path.cwd()}/run_on_node.sh"

print(command)
os.system(command)
