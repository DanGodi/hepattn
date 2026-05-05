import os
import sys
from pathlib import Path

ncpus = "48"
ngpus = "4"
mem = "200gb"


run_eval = sys.argv[1] == "eval" if len(sys.argv) > 1 else False

walltime = "32:00:00" if run_eval else "72:00:00"

command = f"qsub -o {Path.cwd()}/logs/output_fp32.log"
command += f" -e {Path.cwd()}/logs/error_fp32.log"
command += f" -q gpu -N glow_atlas -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=0.1"
# command += f" -v COMET_EXP_ID=594fc1a10d2343089f4115fdf80e3627"
if run_eval:
    command += f" {Path.cwd()}/eval_on_node_fevt.sh"
else:
    command += f" {Path.cwd()}/run_on_node.sh"

print(command)
os.system(command)
