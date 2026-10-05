#!/usr/bin/env bash
# Explicit actions only. No billing, provisioning, credentials, or automatic E2.
set -euo pipefail
profile=${1:?Usage: firm_gpu_run.sh E1A|E1B preflight|train|resume|reload|collect [checkpoint]}
action=${2:?Missing action}
case "$profile" in
  E1A) config=configs/firm3_e1a_reviewed.json ;;
  E1B) config=configs/firm3_e1b_qwen35.json ;;
  *) exit 2 ;;
esac
cd "$(dirname "$0")/.."
firm_venv=${FIRM_VENV:-.venv-gpu-$profile}
firm_python=$firm_venv/bin/python
firm_run=${FIRM_RUN_DIR:?Set FIRM_RUN_DIR to a persistent, dedicated run directory}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
case "$action" in
  preflight) "$firm_python" scripts/train_firm_qlora.py --config "$config" --out "$firm_run" --dry-run ;;
  train) "$firm_python" scripts/train_firm_qlora.py --config "$config" --out "$firm_run" --dry-run
    mkdir -p "$(dirname "$firm_run")"
    if [[ -e "$firm_run.software-lock.txt" ]]; then printf '%s\n' 'Software snapshot exists; use resume or a new run directory.'; exit 2; fi
    "$firm_python" -m pip freeze > "$firm_run.software-lock.txt"
    "$firm_python" scripts/train_firm_qlora.py --config "$config" --out "$firm_run"
    cp "$firm_run.software-lock.txt" "$firm_run/software-lock.txt" ;;
  resume) checkpoint=${3:?Specify an existing checkpoint directory}
    "$firm_python" scripts/train_firm_qlora.py --config "$config" --out "$firm_run" --resume-from-checkpoint "$checkpoint" ;;
  reload) "$firm_python" scripts/validate_firm_adapter.py --run "$firm_run" ;;
  collect) "$firm_python" scripts/package_firm_gpu.py --collect "$firm_run" --out "$firm_run/../$profile-artifacts.tar.gz" ;;
  *) exit 2 ;;
esac
printf '%s\n' 'Action complete. Collect/check artifacts, then stop the rented GPU machine through your provider.'
