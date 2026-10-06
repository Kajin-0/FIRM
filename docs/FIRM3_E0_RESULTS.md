# FIRM 3 E0 measurements and protocol

**Exact pinned Qwen3.5-9B E0: RUN on 2026-10-06.** The pinned Hugging Face revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` was served in BF16 on one NVIDIA L4 with vLLM 0.31.0, context 8192, temperature 0, seed 42, tools off and thinking explicitly off. The complete five-set standard-track run is preserved under `evals/results/E0-qwen35-pinned/`; all 78 requests returned terminal responses with zero backend generation errors. This closes the exact-revision E0 gap for Qwen3.5-9B only; other pinned candidates remain unmeasured.

Primary structured results are poor. Numeric pilot: **1/22 numerical quantities (4.55%)**, **13/22 accepted units (59.09%)**, 3/8 strict structured parse failures, no truncations. Public science development v2: **4/106 numerical quantities (3.77%)**, **41/106 accepted units (38.68%)**, 25/40 strict parse failures, one output-budget truncation, and zero backend generation errors. Legacy keyword proxies are core **0.363**, boundary **0.440**, expert **0.412**; these remain contaminated coverage proxies, not scientific scores. Raw predictions, generation manifests, scorer outputs, server metadata and vLLM log are preserved.

The exact pinned BF16 result is numerically worse than the exploratory local `qwen3.5:9b` Q4 compact track (pilot 3/22; development 9/106), but the local artifact's upstream Hugging Face revision is unknown. Therefore this reversal must **not** be attributed to quantization or precision alone; revision/template/runtime differences are confounded. The exact pinned result nevertheless shows that Qwen3.5-9B, as executed under the declared standard track, is not a strong quantitative infrared baseline.

## Actual local model and limitations

An existing Ollama 0.32.15 service at 127.0.0.1:11434 has `qwen3.5:9b`, Q4_K_M, 9.7B advertised parameters. Installed artifact digest: `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`. Its **upstream HF revision is unknown**. A server-residency observation records size_vram=0: CPU inference, context 8192. That observation is not peak memory. Runner metadata preserves implementation version, artifact digest, model info, renderer/template hashes and default parameters; raw responses retain token counts, reasoning content, finish reason and timings.

The runner refuses absent/cloud Ollama models and never pulls weights. Gold values, units, solutions and manual rubrics are withheld; only prompts and requested quantity names are sent. No scientific tools or external retrieval are enabled. Backend temperature 0, seed 42, presence penalty 0, top_p 1; native Ollama top_k 0; thinking explicitly off. A seed plus temperature zero does not guarantee identical floating-point execution across hardware/server versions.

## Preserved initial diagnostic track

`evals/results/E0-local-qwen35-Q4/` uses original verbose protocol v1, 1024 output tokens. The complete eight-case pilot scored **6/22 numerical quantities (27.27%)**, units 12/22 (54.55%), three strict JSON parse failures, all three corresponding to token-limit truncation. Quantity accuracy includes missing quantities as failures, not a fabricated score over completed answers. `numeric_pilot_score_v2.json` is the final rescore; the earlier score remains preserved. The rescore acknowledges nV-to-V unit handling rather than changing predictions or gold.

Manual analytic inspection found incorrect composition/warm-cutoff behavior; NEP density confused with RMS power and incorrect Jones values; colored-PSD integration underestimated. Beer-Lambert thickness/absorption and the one-pole ENBW/RMS noise pair were numerically correct. Three truncated cases cannot support a complete-answer quality judgment. The 10-case core diagnostic was run under the same verbose protocol; truncation is retained. The initial development diagnostic was interrupted after repeated truncation, with its partial predictions and interruption reason preserved.

The verbose protocol was unsuitable for the local budget. A separate `scientific-compact-v2` protocol puts quantities first and requests a <=200-word explanation. This is a declared protocol change, not selective correction of a model's failed answers. Local reruns have a common 2048-token budget; the future primary GPU standard track uses 4096. Do not pool old/new tracks or infer training deltas from them.

## Completed compact pilot

All eight requests completed without token-limit truncation. Final strict scoring is **3/22 (13.64%)**, units **12/22 (54.55%)**, **3/8 structured parse failures**: prose outside JSON, an invalid JSON escape, and a wrong object schema. Missing parsed quantities fail. Raw predictions and `numeric_pilot_manual_review.json` are retained in `evals/results/E0-local-qwen35-Q4-compact-v2/`. Use `numeric_pilot_score_v2.json`; the first score remains preserved.

The lower score is not a training delta: the protocol/budget changed and no training occurred. All eight outputs were individually inspected. They show composition inversion and temperature-sign errors, cm/m/Jones scaling errors, tau/fc inconsistency, PSD integration failures and invalid causal claims about surface traps. Critically, Beer-Lambert structured quantities pass while the explanation converts 0.003 cm to **3 um**, a factor-of-ten error (correct: 30 um). Numeric accuracy alone would miss this scientific failure. Manual rubric observations are qualitative, not a calibrated human physical score.

## Completed public development and legacy diagnostics

Development v2: **40 attempts, 39 terminal replies, 9/106 numerical quantities (8.49%)**, **65/106 accepted units (61.32%)**, **16/40 strict parse failures**, **two output-budget truncations** (018/019). Case 013 is a backend failure: Ollama returned `done=false`, no finish reason and no token counts despite a non-streaming request. Its partial reply is preserved and its quantities fail; the reported 40/40 prediction coverage means attempted records, not 40 complete answers. Use `science_dev_score_v2.json` and `results_manifest_v1.json` for final scoring and terminal coverage respectively.

All structured quantities and response excerpts were inspected, with focused equation/measurement checks recorded in `science_dev_manual_review.json`. This is qualitative AI review, not calibrated full human rubric grading. Failures include treating PSD as ASD, incorrect 2*pi lifetime factors, omission of the square-wave 2/pi fundamental, projected versus unprojected solid angle, gain/QE confusion, and inconsistent numerical/prose answers. Case 020 claims a passive lowpass increases variance over the same integration support, violating its attenuation bound. Case 031 predicts 0.98 absorbed incident fraction despite a 0.20 reflection loss, violating the 0.80 upper bound. No physical score is invented from these observations.

The three preserved legacy sets each have **10/10 terminal replies**, no compact-track truncations. Keyword coverage proxies are **core 0.307; boundary 0.415; expert 0.445**. Text scoring output and selected-response findings are preserved in `legacy_diagnostic_review.json`. These contaminated proxies are not numerical accuracy or a base-model ranking. For example, expert 004 incorrectly requires a demodulated lock-in time constant ten times shorter than the chop period; expert 005 divides current noise by voltage responsivity. Expert 001 achieves keyword proxy 1.0 despite unsupported mechanism attribution.

Four mathematically equivalent unit spellings observed in raw outputs were added to the explicit grading whitelist: seconds, V/A, reordered radiance denominator, and cm*Hz^{1/2}/W. Rescoring changes accepted-unit counts only (pilot 11->12; development 61->65); **numerical scores remain 3/22 and 9/106**. Old scores and predictions remain intact, and v2 reports record the grader SHA. Unsupported/wrong dimensions still fail. Unit scoring is limited to this audited vocabulary.

Two runner bookkeeping defects were repaired for future runs with regression tests: prose-only legacy answers have no JSON contract, and nonterminal replies now retain a generation-error flag and `complete_with_generation_errors` run status. Historical records are untouched; their 30 legacy `structured_parse_error` flags are inapplicable, not 30 compliance failures. Exact executed runner source snapshots matching recorded SHA-256 hashes are archived under `protocol_sources/`; Git identifiers at run start differed while metadata changes were being developed, and the source hashes make that explicit.

The Bash suite wrapper exited 2 with unexpected EOF after all requested inference manifests completed because it was edited while Bash was still reading it. I verified every expected ID, eval hash, prediction hash and source snapshot; the current wrapper passes `bash -n`. This orchestration mistake did not trigger selective retries or replacement of failed answers. Never edit a live launch wrapper.

The local artifact remains unsuitable for same-revision comparison, but the exact pinned Qwen3.5-9B GPU baseline has now been measured. It is not sufficient to select a winner among the pinned candidate set because the other candidates remain unmeasured. **E2 remains NO-GO.**

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

The Qwen3.5-9B exact GPU run used a **separate serving environment**, as intended. vLLM 0.31.0 resolved to Torch 2.13.0+cu130 and Transformers 5.17.0 on NVIDIA L4 driver 580.178.04; mistral_common 1.12.0 was installed. The actual launch used BF16, context 8192, eager mode, `--reasoning-parser qwen3`, GPU-memory utilization 0.95 and `--skip-mm-profiling`; exact metadata is preserved with the results. Do not mix this serving environment with the validated training environment.

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
