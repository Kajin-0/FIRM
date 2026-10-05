# FIRM 3 E0 measurements and protocol

**Pinned Hugging Face candidate comparison: NOT RUN.** The VPS has no usable CUDA GPU or already-running server with the pinned HF revisions. No endpoint was provisioned and no paid GPU/cloud model was invoked. Local installed Qwen3.5 GGUF measurements below are real, separately labeled exploratory results; they do not satisfy the same-revision E0 gate for adaptation.

## Actual local model and limitations

An existing Ollama 0.32.15 service at 127.0.0.1:11434 has `qwen3.5:9b`, Q4_K_M, 9.7B advertised parameters. Installed artifact digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`. Its **upstream HF revision is unknown**. A server-residency observation records size_vram=0: CPU inference, context 8192. That observation is not peak memory. Runner metadata preserves implementation version, artifact digest, model info, renderer/template hashes and default parameters; raw responses retain token counts, reasoning content, finish reason and timings.

The runner refuses absent/cloud Ollama models and never pulls weights. Gold values, units, solutions and manual rubrics are withheld; only prompts and requested quantity names are sent. No scientific tools or external retrieval are enabled. Backend temperature 0, seed 42, presence penalty 0, top_p 1; native Ollama top_k 0; thinking explicitly off. A seed plus temperature zero does not guarantee identical floating-point execution across hardware/server versions.

## Preserved initial diagnostic track

`evals/results/E0-local-qwen35-Q4/` uses original verbose protocol v1, 1024 output tokens. The complete eight-case pilot scored **6/22 numerical quantities (27.27%)**, units 12/22 (54.55%), three strict JSON parse failures, all three corresponding to token-limit truncation. Quantity accuracy includes missing quantities as failures, not a fabricated score over completed answers. `numeric_pilot_score_v2.json` is the final rescore; the earlier score remains preserved. The rescore acknowledges nV-to-V unit handling rather than changing predictions or gold.

Manual analytic inspection found incorrect composition/warm-cutoff behavior; NEP density confused with RMS power and incorrect Jones values; colored-PSD integration underestimated. Beer-Lambert thickness/absorption and the one-pole ENBW/RMS noise pair were numerically correct. Three truncated cases cannot support a complete-answer quality judgment. The 10-case core diagnostic was run under the same verbose protocol; truncation is retained. The initial development diagnostic was interrupted after repeated truncation, with its partial predictions and interruption reason preserved.

The verbose protocol was unsuitable for the local budget. A separate `scientific-compact-v2` protocol puts quantities first and requests a <=200-word explanation. This is a declared protocol change, not selective correction of a model's failed answers. Local reruns have a common 2048-token budget; the future primary GPU standard track uses 4096. Do not pool old/new tracks or infer training deltas from them.

## Completed compact pilot

All eight requests completed without token-limit truncation. Strict scoring is **3/22 (13.64%)**, units **11/22 (50%)**, **3/8 structured parse failures**: prose outside JSON, an invalid JSON escape, and a wrong object schema. Missing parsed quantities fail. Raw predictions and `numeric_pilot_manual_review.json` are retained in `evals/results/E0-local-qwen35-Q4-compact-v2/`.

The lower score is not a training delta: the protocol/budget changed and no training occurred. All eight outputs were individually inspected. They show composition inversion and temperature-sign errors, cm/m/Jones scaling errors, tau/fc inconsistency, PSD integration failures and invalid causal claims about surface traps. Critically, Beer-Lambert structured quantities pass while the explanation converts 0.003 cm to **3 um**, a factor-of-ten error (correct: 30 um). Numeric accuracy alone would miss this scientific failure. Manual rubric observations are qualitative, not a calibrated human physical score.

The public development and complete 30-case legacy runs are in progress at this documentation checkpoint; their results will be recorded separately when complete. This local artifact in non-thinking mode is insufficient evidence to select or reject a pinned HF candidate, and it gives no reason to start E2.

## Comparison tracks and frozen assets

| Track | Budget / context | Thinking | Role |
|---|---|---|---|
| E0-standard-compact-v2 | 4096 / 8192 | explicitly off | Primary comparison among models supporting non-thinking |
| E0-reasoning-compact-v2 | 8192 / 16384 | explicitly on or declared native always-on | Separate comparison including reasoning-only models |
| E0-local-standard-Q4-compact-v2 | 2048 / 8192 | off | This VPS's installed quantized artifact only |

Use the same prompt/protocol, token budget, temperature, seed, precision and tool-free policy within a comparison track. Record actual chat template, server implementation/version, quantization/precision, GPU, context, thinking controls and complete launch command. Native always-on models do not silently enter the standard track. Token budgets include generated reasoning; reasoning_content and finish reasons are retained. JSON schema forcing is not enabled. Parse failures and truncation are reported alongside numerical accuracy.

Report separately: **pilot** (8 cases/22 quantities), **public science development v2** (40/106), and **legacy diagnostic** (three sets of 10). Legacy is contaminated and its keyword proxy cannot select a base model. Manual physical/derivation/diagnosis/citation metrics remain unscored unless individual review records exist. The development suite is public and reviewed analytically by AI, not a permanent hidden benchmark.

## Candidate revisions and ready-to-run GPU commands

Exact revisions/licenses were reverified from publisher metadata; `configs/firm3_e0_models_v1.json` pins them. No candidate is declared the winner. Qwen3.5-9B is the first technical smoke target; Ministral 14B Reasoning and Qwen3.8-27B remain unmeasured challengers; Qwen3-8B is the conventional-stack control. Publisher sources: [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B), [Ministral 14B Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512), [Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

On an already authorized GPU machine, use a **separate serving environment**. vLLM 0.31.0 and mistral_common 1.12.0 are current pinned release candidates; their registry contains the native Qwen3.5 and Mistral3 classes. Actual CUDA server compatibility/resolution remains NOT RUN; do not mix vLLM's Torch dependencies with the validated training environment. Pin and record the full resolved serving environment before measurements. The publisher's original Qwen instructions mention nightly vLLM; no unpinned nightly is installed here.

```bash
python3.13 -m venv .venv-e0
.venv-e0/bin/python -m pip install vllm==0.31.0 mistral_common==1.12.0
.venv-e0/bin/python -m pip check
.venv-e0/bin/python -m pip freeze > runs/E0-server-software.txt
.venv-e0/bin/vllm serve Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size 1 --seed 42 --generation-config vllm --reasoning-parser qwen3 --gpu-memory-utilization 0.8 --enforce-eager
```

Before invoking the runner, write `runs/E0-server-metadata.json` with actual server/package versions, the exact command and pinned revision, BF16 precision, GPU/driver, context 8192, thinking-off control, actual chat-template hash and no tools. Then:

```bash
FIRM_E0_SERVER_METADATA=runs/E0-server-metadata.json bash scripts/run_firm_e0_suite.sh Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a evals/results/E0-qwen35-pinned
python3 scripts/eval_firm_science.py --eval evals/firm_numeric_pilot_v1.jsonl --pred evals/results/E0-qwen35-pinned/firm_numeric_pilot_v1_predictions.jsonl --out evals/results/E0-qwen35-pinned/pilot_score.json
python3 scripts/eval_firm_science.py --eval evals/firm_science_dev_v2.jsonl --pred evals/results/E0-qwen35-pinned/firm_science_dev_v2_predictions.jsonl --out evals/results/E0-qwen35-pinned/science_dev_score.json
python3 scripts/eval_firm_keywords.py --eval evals/firm_core_eval.jsonl --pred evals/results/E0-qwen35-pinned/firm_core_eval_predictions.jsonl
```

Use the other two preserved legacy files with the same keyword command, and manually inspect responses against their rubrics. The suite does not give the legacy score equal status with numerical evaluation. Resume requires the same recorded code/Git/package identity and run arguments; use the recorded checkout rather than silently resuming after edits.

Other launch templates, **not executed**:

```bash
.venv-e0/bin/vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size 1 --seed 42 --generation-config vllm --reasoning-parser qwen3 --gpu-memory-utilization 0.85 --enforce-eager
.venv-e0/bin/vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 --tensor-parallel-size 1 --seed 42 --generation-config vllm --reasoning-parser qwen3 --gpu-memory-utilization 0.8 --enforce-eager
.venv-e0/bin/vllm serve mistralai/Ministral-3-14B-Reasoning-2512 --revision 51f9210f3cd20f3452a80d5819d15dc61cc50630 --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 16384 --tensor-parallel-size 1 --seed 42 --generation-config vllm --tokenizer-mode mistral --config-format mistral --load-format mistral --reasoning-parser mistral --enforce-eager
```

For the reasoning comparison, relaunch Qwen with context 16384 too, set FIRM_E0_TOKENS=8192, FIRM_E0_CONTEXT=16384, FIRM_E0_THINKING=on and FIRM_E0_TRACK=E0-reasoning-compact-v2. For Ministral's native reasoning control use server-default only with explicit native-always-on metadata. A100/H100 80 GB is the planning class for 27B BF16 inference; actual fit is a server gate, not a measured claim. No multi-GPU run is required merely to verify these commands.
