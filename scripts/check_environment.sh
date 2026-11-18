#!/usr/bin/env bash
# Print a whitelist of environment facts. Do not dump the process environment.
set +x
set +v
set -u
echo "hostname=$(hostname)"
echo "machine=$(uname -m)"
echo "kernel=$(uname -r)"
echo "cpu_count=$(nproc)"
if [[ -x /home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python ]]; then
  echo "analysis_python=$(/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python --version 2>&1)"
else
  echo "analysis_python=missing"
fi
echo "sinfo=$(command -v sinfo || true)"
echo "sbatch=$(command -v sbatch || true)"
sinfo -h -o '%P %a %D %c %G' 2>/dev/null | head -n 5 || true
echo "slurm_user=$(id -un)"
sacctmgr -nP show assoc user="$(id -un)" format=Account,QOS 2>/dev/null | head -n 5 || true
