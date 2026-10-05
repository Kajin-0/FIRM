#!/usr/bin/env bash
# Calls the existing runner for five preserved assets; never starts a server.
set -euo pipefail
firm_model=${1:?model}
firm_revision=${2:?immutable revision or local artifact digest}
firm_out=${3:?new output directory}
firm_sets=${4:-all}
cd "$(dirname "$0")/.."
case "$firm_sets" in
  all) assets=(firm_numeric_pilot_v1 firm_science_dev_v2 firm_core_eval firm_domain_boundary_eval firm_v2_expert_eval) ;;
  pilot-legacy) assets=(firm_numeric_pilot_v1 firm_core_eval firm_domain_boundary_eval firm_v2_expert_eval) ;;
  *) exit 2 ;;
esac
for asset in "${assets[@]}"; do
  extra=()
  if [[ -n "${FIRM_E0_SERVER_METADATA:-}" ]]; then extra+=(--server-metadata "$FIRM_E0_SERVER_METADATA"); fi
  python3 scripts/run_firm_baseline.py --eval "evals/$asset.jsonl" \
    --model "$firm_model" --model-revision "$firm_revision" \
    --base-url "${FIRM_E0_URL:-http://127.0.0.1:8000/v1}" --backend "${FIRM_E0_BACKEND:-openai}" \
    --thinking "${FIRM_E0_THINKING:-off}" --track "${FIRM_E0_TRACK:-E0-standard-compact-v2}" \
    --protocol scientific-compact-v2 --max-tokens "${FIRM_E0_TOKENS:-4096}" \
    --context-length "${FIRM_E0_CONTEXT:-8192}" --out "$firm_out/${asset}_predictions.jsonl" "${extra[@]}"
done
