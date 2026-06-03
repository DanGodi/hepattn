import os
from datetime import datetime
from pathlib import Path

ncpus = "48"
ngpus = "4"
mem = "200gb"
walltime = "72:00:00"

logs_dir = Path.cwd() / "logs"
logs_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

command = f"qsub -o {logs_dir}/pbs_out_{stamp}.log"
command += f" -e {logs_dir}/pbs_err_{stamp}.log"
command += f" -q gpu -N glow_atlas -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=0.1,gputype=A6000"
command += f" {Path.cwd()}/run_on_node.sh"

print(command)
os.system(command)
