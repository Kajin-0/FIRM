# AGENTS.md — FIRM project operating context

## Mission

FIRM (Focused Infrared Research Model) exists to produce the best practical **small, local language model for infrared photonics and enabling semiconductor science**.

The target is not merely a good benchmark score and not a general chatbot with an infrared system prompt. Optimize jointly for:

1. **Scientific accuracy** — especially infrared detector materials, devices, radiometry, noise, semiconductor physics, fabrication, characterization, and quantitative work.
2. **Robustness** — handle paraphrases, typos, terse prompts, adjacent semiconductor science, underdetermined questions, and adversarial misconceptions without false refusals or confident fabrication.
3. **Quantitative reliability** — equations, units, dimensional analysis, PSD/ASD distinctions, conversions, assumptions, and numerical answers must survive held-out grading.
4. **Useful specialist scope** — answer valid in-domain/supporting-science questions directly; decline genuinely unrelated requests without allowing refusal behavior to become a shortcut for accuracy.
5. **Laptop-class deployment efficiency** — minimize memory, time-to-first-token, answer latency, and decode cost while preserving quality. CPU/Ollama operation is a first-class target; laptop GPU acceleration is optional, not required for usefulness.
6. **Reproducibility and provenance** — exact model revisions, data hashes, code revisions, raw predictions, quantization lineage, and evaluation conditions must be recoverable.

The long-term product goal is the highest-quality specialist model that remains realistically runnable on a normal laptop. Parameter count is a constraint, not a prestige metric. Prefer a smaller/faster model when held-out scientific quality is equivalent.

## Current compact-model direction

The active compact family is based on the exact pinned **Qwen/Qwen3.5-4B** revision:

`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

FIRM-4B v1 was the first real specialist fine-tune. It showed useful scope/style learning but had serious factual and robustness failures documented in `docs/FIRM4B_V1_FINDINGS_AND_V2_PLAN.md`.

FIRM-4B v2.1 is the current substantive experiment. Its frozen training source is commit:

`c6e18c0617d9462f8a2593929d57f7c26aff2ba0`

At that commit the v2.1 curriculum/trainer uses:
- 4,973 train examples
- 432 validation examples
- 157 canonical fact targets
- 50 diagnostic-reasoning targets
- 16 taxonomy targets
- 20 explicit misconception-correction targets
- 2,355 quantitative examples
- <1% refusal examples
- exact private stress prompts and private benchmark excluded
- BF16 LoRA, rank 16 / alpha 32
- LR 1.5e-5
- effective batch 4
- sequence length 1024
- 1,250 optimizer steps, approximately one effective pass

Do **not** infer that a run completed successfully merely because this file names the experiment. Inspect `runs/gpu-results/`, run metadata, checkpoint/final artifacts, and logs.

## Scientific behavior

Authoritative scope is `docs/FIRM3_SPECIALIST_CHARTER.md`.

Within scope, prefer answers that:
- state assumptions when material,
- use correct governing equations,
- define variables and units,
- distinguish measured facts from models/hypotheses,
- identify underdetermination instead of inventing missing measurements,
- connect physics to observable measurement consequences,
- remain concise unless detail is necessary.

High-priority failure modes include:
- MCT/CZT/InGaAs/InSb identity errors,
- incorrect HgCdTe substrate guidance,
- detector-taxonomy errors,
- D* / NEP / PSD / ASD / NASD confusion,
- cutoff-bandgap-composition errors,
- false refusal caused by typos or missing infrared keywords,
- invented organizations, citations, materials, formulas, measurements, or model provenance,
- overconfident mechanism attribution from insufficient evidence.

Do not hard-code answers or create prompt-routing tricks to inflate evaluation scores.

## Evaluation contract

Never call a candidate improved from training loss alone.

Every meaningful candidate must be compared against relevant predecessors and the untouched pinned base using held-out data. Keep these axes separate:

- factual/domain correctness,
- numerical/unit correctness,
- diagnostic reasoning,
- in-domain acceptance / false-refusal rate,
- out-of-domain refusal precision,
- misconception resistance,
- hallucination/fabrication rate,
- model-identity honesty,
- memory footprint,
- model file size,
- load time / time-to-first-token,
- prompt-processing throughput,
- decode tokens/s,
- end-to-end answer latency.

For laptop deployment, report hardware, backend, quantization, context length, thread/GPU settings, prompt length, generated-token count, and whether thinking/reasoning mode is enabled. Prefer p50/p95 latency when enough trials exist.

Quantization is part of the model-release experiment. Validate exact adapter/base lineage before conversion and compare the quantized artifact against the unquantized fine-tuned model for quality regressions.

## Benchmark integrity

Benchmark contamination invalidates the result.

- Do not train on exact user stress-test prompts.
- Do not import the private benchmark into curriculum builders or teacher prompts.
- Do not tune manually against private-test answers.
- Public development suites may guide engineering, but do not mislabel them as unbiased hidden tests.
- Preserve exact raw model outputs whenever practical.
- Score scope separately from correctness so refusing everything cannot look accurate.
- Generated teacher explanations are not ground truth without independent verification.

## Data quality

Prefer fewer high-confidence examples over large quantities of weak synthetic prose.

Training targets should be:
- physically correct,
- dimensionally correct,
- explicit about assumptions where needed,
- original/paraphrased rather than copied source text,
- traceable to evidence or an independently verified derivation,
- diverse enough that the model learns concepts rather than answer strings.

Programmatically generated quantitative examples are valuable only when their oracles and units are independently checked. Repetition can overweight narrow behaviors.

## Architecture and efficiency strategy

The default compact target is approximately 3–4B parameters unless evidence shows another size is a better laptop-quality Pareto point.

Current Qwen3.5 training adapts only language-model MLP projections. Vision, embeddings, LM head, attention/DeltaNet state machinery, and other frozen parameters must remain frozen unless a deliberate ablation changes that contract.

Future improvements should be evidence-driven. Possible levers include:
- better curriculum coverage and data quality,
- LoRA target/rank/LR ablations,
- knowledge-preserving specialist tuning,
- small amounts of verified reasoning/diagnostic data,
- distillation from stronger teachers with independent answer verification,
- retrieval/tools where they beat memorization,
- validated low-bit quantization,
- inference-kernel/backend optimization.

Do not increase parameter count, reasoning-token budget, context size, or runtime complexity merely to make the model appear smarter. Compare quality gained per memory/latency cost.

## Engineering/reproducibility rules

Before substantial changes:
1. inspect `git status`, branch, and HEAD;
2. read this file and the relevant current experiment docs/manifests;
3. run the applicable tests/preflights;
4. preserve old outputs and use new run paths;
5. pin model revisions and record hashes;
6. keep evaluation and training artifacts separable.

Core checks normally include:

```bash
python3 -m unittest discover -s tests -q
python3 -m compileall -q scripts tests
git diff --check
```

For the active compact trainer, preserve exact train/validation hashes, base revision, adapter boundaries, frozen-weight checks, and checkpoint identity.

## GPU/cloud safety

Cloud GPU runs are bounded experiments, not open-ended infrastructure.

- Do not create long-lived service-account keys.
- Do not place the user's personal Google credentials on a training VM.
- Do not give a VPS broad Google Cloud access beyond the explicit temporary user flow.
- Keep training VMs isolated from unrelated projects/resources.
- Use provider-enforced runtime/deletion guards when available.
- Evacuate checkpoints/final weights before deletion.
- Audit instances, disks, addresses, and reservations after cleanup.
- Revoke temporary cloud credentials after cleanup is verified.
- Never increase quota, enable billing, change billing permissions, or broaden spending without explicit user approval.

## Context recovery for a new agent

Read in this order:

1. `AGENTS.md`
2. `README.md`
3. `docs/FIRM3_SPECIALIST_CHARTER.md`
4. `docs/FIRM4B_V1_FINDINGS_AND_V2_PLAN.md`
5. current trainer/builder/tests for the active experiment
6. current `runs/gpu-results/` metadata/logs
7. evaluation scripts and frozen eval manifests
8. `git log --oneline`, `git status`, and the remote branch state

Older FIRM3 documents are important provenance but can describe superseded pre-4B states. When prose conflicts with executable manifests/tests and a later committed experiment, verify chronology rather than silently assuming the oldest document is current.

## Decision principle

Choose the next experiment by the largest expected improvement in **held-out specialist quality per unit of model size, memory, latency, engineering complexity, and training cost**.

The final deliverable should be a compact specialist that is measurably better than its base model, scientifically trustworthy within its stated scope, robust to ordinary user phrasing, and fast enough that running it locally on a laptop is practical.
