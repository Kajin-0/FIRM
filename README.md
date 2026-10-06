# FIRM: Focused Infrared Research Model

FIRM is a local, domain-specialized language model project for infrared photodetectors, with emphasis on HgCdTe/MCT, detector physics, noise, radiometry, cryogenic testing, lock-in measurements, and empirical troubleshooting.

## Immediate objective

Develop FIRM through reviewed scientific data, contamination-aware evaluation and controlled experiments. Scientific review and a native modern-model CPU integration gate are complete. A local Q4 base-model diagnostic was measured; no domain-adapted FIRM 3 weights exist. Full-size GPU validation and an exact-revision baseline remain required before substantive tuning.

Start with [the handoff](docs/FIRM3_HANDOFF.md). Detailed findings, architecture and experiments are in [FIRM3_AUDIT](docs/FIRM3_AUDIT.md), [FIRM3_SPEC](docs/FIRM3_SPEC.md) and [FIRM3_TRAINING_PLAN](docs/FIRM3_TRAINING_PLAN.md).

Current evidence: [scientific review](docs/FIRM3_SCIENTIFIC_REVIEW.md), [actual E0 diagnostics](docs/FIRM3_E0_RESULTS.md), [GPU execution runbook](docs/FIRM3_GPU_RUNBOOK.md). **E2 is NO-GO** until the recorded scientific, rights, baseline, GPU and spending gates pass.

## Target behavior

FIRM should answer technical questions using:

1. assumptions,
2. governing equations,
3. defined variables and units,
4. physical interpretation,
5. measurement consequences,
6. empirical caveats and failure modes.

## Repository layout

```text
FIRM/
├── configs/                 # Training configuration files
├── data/
│   ├── curation/            # Preserved numbered rewrite transformations
│   ├── processed/           # Legacy SFT exports; local candidate splits
│   ├── manifests/           # Source hashes, review exclusions and experiment schemas
│   ├── reviews/             # Pinned individual reviews, corrections and analytic evidence
│   └── audits/              # Dataset audit outputs
├── docs/                    # Dataset/model notes
├── evals/                   # Frozen releases, public development suite and real local results
├── scripts/                 # Audit, conversion, training, evaluation scripts
├── tests/                   # CPU data/leakage/unit/oracle regression checks
└── README.md
```

## Current dataset audit snapshot

Independently reproduced on 2026-10-05, using whitespace-separated words. The historical audit report is preserved, but its length metrics do not reproduce under this convention.

| Metric | Value |
|---|---:|
| Usable rows | 2,532 |
| Required fields missing | 0 |
| Exact duplicate input/output pairs | 0 |
| Exact duplicate inputs | 0 |
| Duplicate outputs | 114 |
| Mean input length | 14.58 words |
| Mean output length | 21.02 words |
| 95th percentile output length | 31 words |
| Max output length | 299 words |

The rewritten 2,532-row corpus averages 33.30 response words, but most remains unreviewed. All 150 templated D* errors were individually confirmed and corrected in a versioned transformation; originals remain intact. Corrected variants remain quarantined because their family is eval-equivalent. Long-form sources still contain unresolved unit, arithmetic and measurement-model errors. Full original statistics: [audit snapshot](data/audits/firm3_2026-10-05/README.md).

Active [reviewed seed v3](data/processed/firm3_reviewed_seed_v3/manifest.json): **45 train / 1 validation / 1 test / 159 quarantine / 725 excluded / 1,650 unreviewed**. Only accepted/corrected eligible rows can train. The tiny validation/test partitions are plumbing checks. Reviews are AI analytical checks; source rights and human expert signoff remain unresolved. This is a smoke dataset, not a sufficient long-form scientific SFT corpus.

## Current model and experiment strategy

```text
E0 lightweight baseline: Qwen/Qwen3.5-9B
E0 medium challenger: Ministral 3 14B Reasoning
E0 research baseline: Qwen/Qwen3.8-27B
E1A reviewed legacy smoke: pinned Qwen/Qwen3-0.6B, 10 NF4 optimizer steps
E1B native modern smoke: pinned Qwen/Qwen3.5-9B, 2 BF16 LoRA steps, 8 rows
Future architecture: measured DAPT -> expert SFT -> correctness -> tools/RAG -> quantization
```

Selection is provisional. Qwen3-8B remains the conventional-stack control. The native Qwen3.5 text-only profile uses its conditional-generation class and 96 language MLP LoRA targets, preserving vision/hybrid attention. A tiny native CPU fixture passed forward/backward, masks, frozen weights, checkpoint/resume and exact-logit reload. Actual 9B CUDA/BF16 fit and modern NF4 are unvalidated; modern NF4 is currently refused. See the runbook for evidence and limits.

## Reproduce checks without a GPU

The new data and scoring tools use the Python standard library. Existing CSV builders can use `requirements-data.txt`; installing the full GPU stack is unnecessary for an audit.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/audit_firm_data.py --out data/audits/local_snapshot --date 2026-10-05
```

For actual token counts, install Transformers 4.57.6 and Jinja2 in an isolated environment and use the pinned tokenizer. The VPS already has an ignored `.venv-audit` with saved tokenizer files:

```bash
.venv-audit/bin/python scripts/audit_firm_data.py \
  --out data/audits/local_token_snapshot --date 2026-10-05 \
  --tokenizer Qwen/Qwen3.5-9B \
  --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --tokenizer-files .venv-audit/firm-tokenizer --local-files-only
```

Fresh clones can omit `--tokenizer-files` after explicitly downloading that tokenizer revision. No weight loading occurs. The shared snapshot records tokenizer-file hashes and package versions.

The following is the **historical unreviewed candidate** builder, retained for reproducibility. Its original output already exists on the VPS; use a NEW directory to regenerate. Actual training refuses unreviewed sources:

```bash
python3 scripts/split_firm_data.py \
  --input data/processed/firm_rewritten_large_sft.jsonl \
  --input data/processed/firm_v2_expert_hgcdte_deep_batch01.jsonl \
  --input data/processed/firm_v2_expert_hgcdte_deep_batch02.jsonl \
  --input data/processed/firm_v2_expert_hgcdte_deep_batch03.jsonl \
  --input data/processed/firm_v2_root_expert_sft.jsonl \
  --out-dir data/processed/firm3_candidate_2026-10-05_v2
```

Historical candidates: 717 train / 65 valid / 106 test, 379 quarantine, 1,314 capped variants, preserved unchanged. Active smoke configurations use reviewed seed v3. Reproduce review/corrections/splits into NEW paths:

```bash
python3 scripts/review_firm_seed.py --out-dir data/reviews/local_review_snapshot \
  --build-seed --require-nonempty --seed-out-dir data/processed/local_reviewed_seed
```

The seven output JSONL hashes reproduce deterministically. Source/review/eval lineage and whole-family grouping are preserved. Automated lexical/structural leakage checks find zero matches across 47 eligible rows; this does not prove semantic independence or unknown base-model pretraining exposure.

## Baseline and numerical grading

Legacy files are preserved under their existing names and frozen by `firm_legacy_v1_manifest.json`; they have training overlaps. The 8-case pilot/22 quantities were independently rederived without errors, so v1 is unchanged. Active `firm_science_dev_v2.jsonl` adds 40 scenarios/106 quantities with independent reference checks. Dev v1 is preserved; v2 replaces one structurally related legacy scenario. Both numeric suites are public development assets with AI analytic review, not a permanent unbiased hidden test.

Actual local Ollama Qwen3.5 Q4_K_M, upstream revision unknown, thinking off: pilot **3/22**, development **9/106** correct. Development has 16 parse failures, two truncations and one nonterminal backend reply. All 30 legacy diagnostics were attempted; they are reported separately. See E0 results for immutable raw predictions, final unit-aware scores and manual findings. The exact pinned HF comparison is **NOT RUN**.

The baseline runner uses an already-running **localhost** compatible server. Dry-run never contacts it or loads a model:

```bash
python3 scripts/run_firm_baseline.py \
  --eval evals/firm_numeric_pilot_v1.jsonl \
  --model Qwen/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --out predictions/E0-qwen35-9b.jsonl --dry-run
```

On an existing server loading that exact revision, remove `--dry-run`. Predictions are append-only and resumable with matching run metadata. Backend revision is operator-declared; verify the server command. No gold values/rubrics are sent. Truncated or malformed structured answers remain visible.

```bash
python3 scripts/eval_firm_science.py \
  --eval evals/firm_numeric_pilot_v1.jsonl \
  --pred predictions/E0-qwen35-9b.jsonl \
  --out predictions/E0-qwen35-9b-score.json
```

Numerical grading expects `{id, answer, quantities: {name: {value, unit}}}`. It checks explicit supported unit conversions and fixed tolerances; physical reasoning/citations remain manual rubric fields. The old keyword scorer is a coverage proxy only.

## Training preflight and reproducibility

```bash
python3 scripts/train_firm_qlora.py --config configs/firm3_e1a_reviewed.json --dry-run
python3 scripts/train_firm_qlora.py --config configs/firm3_e1b_qwen35.json --dry-run
```

These check model profiles, pinned revisions, data/eval hashes and eligibility without loading weights. E1A/E1B CUDA execution is NOT RUN. The provider-neutral archive/bootstrap/resume/reload/collection path is tested after extraction without Git; follow the [GPU runbook](docs/FIRM3_GPU_RUNBOOK.md). No paid resources were provisioned. An authorized bounded E1B GPU smoke is the next technical experiment; passing it does not authorize E2.

Existing rewrite builds remain reproducible through `scripts/apply_rewrite_batches.py`; send outputs to a new temporary directory, not the preserved exports. `freeze_firm_evals.py` and `build_firm_numeric_pilot.py` create versioned artifacts and refuse existing release paths. Older preparation/curation/merge/large-generation scripts remain historical alternatives; consult the audit before using them.

Training runs record git/model revisions, dataset/eval hashes, seed/hyperparameters, software/hardware, checkpoints, losses and benchmark status. Actual GPU runs also measure gradients/frozen state, memory, synchronized timing/tokens, artifact sizes and adapter reload. Repository/dataset licensing and human scientific review remain unresolved; provenance is recorded rather than invented.

## High-priority expansion areas

1. HgCdTe bandgap, composition, cutoff wavelength, and temperature dependence.
2. Intrinsic carrier concentration and dark-current scaling.
3. Photoconductor gain, lifetime, transit time, and frequency roll-off.
4. Johnson, shot, 1/f, and generation-recombination noise.
5. Blackbody radiometry and optical power coupling.
6. Lock-in amplifier measurements, MFLI workflows, and chopped-source testing.
7. LPE, MOCVD, annealing, passivation, and cryogenic packaging.
8. Empirical troubleshooting from IV curves, spectra, noise spectra, and frequency response.
