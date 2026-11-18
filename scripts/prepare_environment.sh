#!/usr/bin/env bash
# Create the project analysis environment on the storage filesystem.
# Does not modify hf-hpc, conda configuration, or shell startup files.
set -euo pipefail
PREFIX=/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis
CACHE=/home/b5ac/freddieyu.b5ac/projects/project_1/cache/pip
if [[ -x "$PREFIX/bin/python" ]]; then
  echo "analysis environment already present: $PREFIX"
  exit 0
fi
mkdir -p "$CACHE" "$(dirname "$PREFIX")"
/home/b5ac/freddieyu.b5ac/miniforge3/bin/python -m venv "$PREFIX"
"$PREFIX/bin/python" -m pip install --cache-dir "$CACHE" \
  "numpy==2.2.6" "pandas==2.3.3" "pyarrow==22.0.0" "pyyaml==6.0.3" \
  "pytest==9.0.2" "matplotlib==3.10.8" "transformers==4.57.5" "tokenizers==0.22.2"
echo "created $PREFIX"
