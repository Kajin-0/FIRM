#!/usr/bin/env bash
# Prepares software on an ALREADY authorized GPU machine; provisions nothing.
set -euo pipefail
profile=${1:?Usage: bootstrap_firm_gpu.sh E1A|E1B}
case "$profile" in
  E1A) firm_python=${FIRM_PYTHON:-python3.11}; req=requirements-training.txt; expected=3.11 ;;
  E1B) firm_python=${FIRM_PYTHON:-python3.13}; req=requirements-modern-gpu.txt; expected=3.13 ;;
  *) exit 2 ;;
esac
cd "$(dirname "$0")/.."
"$firm_python" -c 'import sys; assert ".".join(map(str,sys.version_info[:2])) == sys.argv[1]' "$expected"
firm_venv=${FIRM_VENV:-.venv-gpu-$profile}
"$firm_python" -m venv "$firm_venv"
"$firm_venv/bin/python" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
"$firm_venv/bin/python" -m pip install -r "$req"
"$firm_venv/bin/python" -m pip check
"$firm_venv/bin/python" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
printf '%s\n' 'Software ready. No training started. Use persistent storage and stop the rented machine after collecting artifacts.'
