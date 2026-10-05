# FIRM: Focused Infrared Research Model

FIRM is a local, domain-specialized language model project for infrared photodetectors, with emphasis on HgCdTe/MCT, detector physics, noise, radiometry, cryogenic testing, lock-in measurements, and empirical troubleshooting.

## Immediate objective

Develop FIRM through reviewed scientific data, contamination-aware evaluation and controlled experiments. FIRM 3 infrastructure is established; no FIRM 3 model has been trained or benchmarked yet.

Start with [the handoff](docs/FIRM3_HANDOFF.md). Detailed findings, architecture and experiments are in [FIRM3_AUDIT](docs/FIRM3_AUDIT.md), [FIRM3_SPEC](docs/FIRM3_SPEC.md) and [FIRM3_TRAINING_PLAN](docs/FIRM3_TRAINING_PLAN.md).

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
│   └── audits/              # Dataset audit outputs
├── docs/                    # Dataset/model notes
├── evals/                   # Frozen legacy assets and separate numerical pilot
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

The rewritten 2,532-row corpus averages 33.30 response words, but correctness needs review. All 150 templated D* answers omit the metre-to-centimetre conversion while labeling results Jones. Long-form examples also contain unit, arithmetic and measurement-model errors. The sources remain intact; flagged/eval-equivalent groups are quarantined in new candidate splits. Full source/token/duplicate/leakage statistics: [audit snapshot](data/audits/firm3_2026-10-05/README.md).

## Current model and experiment strategy

```text
E0 lightweight baseline: Qwen/Qwen3.5-9B
E0 medium challengers: Gemma 4 12B / Ministral 3 14B Reasoning
E0 research baseline: Qwen/Qwen3.8-27B
E1 pipeline smoke only: pinned Qwen/Qwen3-0.6B, 10 QLoRA optimizer steps
Future architecture: measured DAPT -> expert SFT -> correctness -> tools/RAG -> quantization
```

Selection is provisional and backed by current publisher sources in the training plan. The pinned legacy causal trainer supports the E1 smoke/control path; current native vision/hybrid candidates need a separately validated training profile.

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

Build candidate splits in a **new** output directory; this command has already been run on the VPS. Existing output directories are refused to preserve artifacts:

```bash
python3 scripts/split_firm_data.py \
  --input data/processed/firm_rewritten_large_sft.jsonl \
  --input data/processed/firm_v2_expert_hgcdte_deep_batch01.jsonl \
  --input data/processed/firm_v2_expert_hgcdte_deep_batch02.jsonl \
  --input data/processed/firm_v2_expert_hgcdte_deep_batch03.jsonl \
  --input data/processed/firm_v2_root_expert_sft.jsonl \
  --out-dir data/processed/firm3_candidate_2026-10-05_v2
```

Current candidates: 717 train / 65 valid / 106 test, 379 quarantine, 1,314 capped variants. They are unreviewed; generated rows are ignored locally, and a tracked snapshot manifest records their hashes. Numeric prompt/response templates and declared document/family identities stay in one partition. Lexical/structural leakage checks plus explicit review exclusions quarantine whole groups; semantic review is still required.

## Baseline and numerical grading

Legacy files are preserved under their existing names and frozen by `firm_legacy_v1_manifest.json`. They have training overlaps. The separate 8-case `firm_numeric_pilot_v1.jsonl` has 22 numerical quantities and public solutions; expert review is pending. Neither is advertised as a clean, comprehensive permanent benchmark.

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
python3 scripts/train_firm_qlora.py --config configs/firm3_e1_smoke.json --dry-run
```

This checks model revision format, data/eval hashes and partition eligibility without importing GPU libraries. An actual E1 requires the pinned CUDA environment, inspected smoke examples, completion-mask verification, checkpoint/resume and adapter reload checks. Follow the training plan before removing dry-run. Serious training was not authorized or launched in this audit session.

Existing rewrite builds remain reproducible through `scripts/apply_rewrite_batches.py`; send outputs to a new temporary directory, not the preserved exports. `freeze_firm_evals.py` and `build_firm_numeric_pilot.py` create versioned artifacts and refuse existing release paths. Older preparation/curation/merge/large-generation scripts remain historical alternatives; consult the audit before using them.

Training runs record git/model revisions, dataset/eval hashes, seed/hyperparameters, software/hardware, checkpoints, losses and benchmark status. No model score is present until actual evaluation. Repository/dataset licensing and human scientific review remain unresolved; provenance is recorded as unknown rather than invented.

## High-priority expansion areas

1. HgCdTe bandgap, composition, cutoff wavelength, and temperature dependence.
2. Intrinsic carrier concentration and dark-current scaling.
3. Photoconductor gain, lifetime, transit time, and frequency roll-off.
4. Johnson, shot, 1/f, and generation-recombination noise.
5. Blackbody radiometry and optical power coupling.
6. Lock-in amplifier measurements, MFLI workflows, and chopped-source testing.
7. LPE, MOCVD, annealing, passivation, and cryogenic packaging.
8. Empirical troubleshooting from IV curves, spectra, noise spectra, and frequency response.
