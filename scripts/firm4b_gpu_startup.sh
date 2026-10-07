#!/usr/bin/env bash
# Self-contained FIRM 4B GPU training bootstrap.
# Executed by GCE as a startup script. No Google credentials, no service account.
# GCE must be configured with a provider-enforced maxRunDuration + DELETE.
set -Eeuo pipefail
umask 022
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PIP_NO_INPUT=1
export FIRM_ROOT=/opt/firmgpu
export FIRM_OUT=/opt/firmgpu/outputs/firm4b-provisional-v1
export HF_HOME=/opt/firmgpu/cache/huggingface
export UV_PYTHON_INSTALL_DIR=/opt/firmgpu/python
export UV_CACHE_DIR=/opt/firmgpu/cache/uv
# NOTE: Exact source commit is injected in a private render step at VM creation.
mkdir -p /opt/firmgpu/outputs /opt/firmgpu/cache /opt/firmgpu/logs
chmod 755 /opt/firmgpu /opt/firmgpu/outputs /opt/firmgpu/logs
exec > >(tee -a /opt/firmgpu/logs/startup.log) 2>&1
say(){ printf '[FIRM %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
status(){
  local state="$1"
  python3 - "$state" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path("/opt/firmgpu/logs/run_status.json")
p.write_text(json.dumps({"status":sys.argv[1],"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},sort_keys=True)+"\n")
PY
}
finish(){
  rc=$?
  if [ "$rc" -eq 0 ]; then status completed; else status "failed_exit_$rc"; fi
  say "Startup script finished with status $rc. Do not confuse preparation with an achieved model."
}
trap finish EXIT
say "Starting pinned compact FIRM experiment."
status provisioning
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
mkdir -p /opt/firmgpu/src
if [ -e /opt/firmgpu/src/FIRM/.git ]; then
  say "Unexpected existing repository, refusing overwrite."
  exit 3
fi
git clone --quiet https://github.com/Kajin-0/FIRM.git /opt/firmgpu/src/FIRM
cd /opt/firmgpu/src/FIRM
git fetch --quiet origin firm3/scientific-validation-e0
git checkout --detach "@FIRM_PINNED_COMMIT@"
test "$(git rev-parse HEAD)" = "@FIRM_PINNED_COMMIT@"
python3 scripts/train_firm_specialist_4b.py --allow-provisional-data --out "$FIRM_OUT" --dry-run
say "Source revision and research-corpus preflight pass."
status installing
python3 -m venv /opt/firmgpu/bootstrap-venv
/opt/firmgpu/bootstrap-venv/bin/python -m pip install --disable-pip-version-check --quiet 'uv==0.9.11'
UV=/opt/firmgpu/bootstrap-venv/bin/uv
"$UV" python install 3.13
"$UV" venv --python 3.13 /opt/firmgpu/venv
"$UV" pip install --python /opt/firmgpu/venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python /opt/firmgpu/venv/bin/python -r requirements-modern-gpu.txt
/opt/firmgpu/venv/bin/python -m pip check || "$UV" pip check --python /opt/firmgpu/venv/bin/python
/opt/firmgpu/venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
print("CUDA_DEVICE",torch.cuda.get_device_name(0), "CUDA",torch.version.cuda,flush=True)
PY
/opt/firmgpu/venv/bin/python -m pip freeze > /opt/firmgpu/logs/software-lock.txt || "$UV" pip freeze --python /opt/firmgpu/venv/bin/python > /opt/firmgpu/logs/software-lock.txt
say "Environment installed."
status preparing
/opt/firmgpu/venv/bin/python scripts/train_firm_specialist_4b.py --allow-provisional-data --out "$FIRM_OUT" --dry-run
status training
# The external timeout precedes the cloud provider enforced delete.
# Periodic intermediate checkpoints remain recoverable even if the run is cut short.
set +e
timeout --signal=INT --kill-after=100s 5400s /opt/firmgpu/venv/bin/python -u scripts/train_firm_specialist_4b.py \
  --allow-provisional-data --data data/processed/firm3_specialist_4b_mix_v1 \
  --out "$FIRM_OUT" --steps 1100 --max-seq-length 1024 \
  > >(tee -a /opt/firmgpu/logs/training.log) 2>&1
rc=$?
set -e
say "Trainer exit status: $rc"
if [ -d "$FIRM_OUT" ]; then
  cp -f /opt/firmgpu/logs/software-lock.txt "$FIRM_OUT/software-lock.txt"
  chmod -R a+rX /opt/firmgpu/outputs
fi
if [ "$rc" -ne 0 ]; then
  status "partial_checkpoint_or_failure_exit_$rc"
  say "Recover latest checkpoint if present; model is not a completed run."
  exit "$rc"
fi
status completed_training_pending_evaluation
say "Training finished; separate exact-base vs adapter evaluation still mandatory."
# Keep VM online until operator/pullwatch confirms checkpoint backup, then delete.
