# FIRM 3 audit — 2026-10-05

## Repository identity and preservation

Verified before editing: Git root `/home/User/FIRM`; remote `git@github.com:Kajin-0/FIRM.git`; branch `main`; local HEAD `17351932a7337946d2017a5aa14c0a7201767096`. `git fetch --all --prune` succeeded. Live `git ls-remote --symref origin HEAD` returned `main` and the same SHA. `git rev-list --left-right --count HEAD...origin/main` returned `0 0`. The initial working tree was clean, with no untracked files. This establishes identity and equality at takeover, rather than inferring them from directory names.

Development branch: `firm3/audit-and-foundation`. No merge, reset, source dataset replacement, existing eval alteration, checkpoint deletion, or paid training occurred. Evidence: `data/audits/firm3_2026-10-05/git_identity.json`. Final branch state and handoff commands are in `FIRM3_HANDOFF.md`.

## Inventory at takeover

55 tracked files: 37 JSONL, 3 CSV, 7 Python scripts, 5 Markdown, 1 requirements file, 1 YAML config, 1 workflow. Approximate Python LOC, including comments/docstrings: 1,323. Top-level distribution: 38 data files, 7 scripts, 3 evals, 2 docs, 1 config, 1 workflow, 3 root files. Full sizes/line counts: audit `inventory.json`, captured before new files were committed. CSV physical lines are not record counts.

| Asset | Records | Role |
|---|---:|---|
| Root original CSV | 2,532 | Original compact seed; preserved |
| Rewritten large CSV | 2,532 | Rewrite-derived alternative |
| Rewritten large SFT JSONL | 2,532 | Same rewritten examples in chat format |
| Expert HgCdTe batches 01 / 02 / 03 | 19 / 10 / 10 | Additional analytical candidate examples |
| Root expert SFT JSONL | 10 | Compact expert seed; alternative/overlapping concepts |
| 28 numbered rewrite files | 815 | Transformations, not 815 additional examples |
| Rewrite overrides JSONL | 8 | Alternative legacy transformation input |
| Core / domain boundary / expert evals | 10 / 10 / 10 | Public prompt/rubric diagnostics |

The root CSV is 771,920 bytes; rewritten SFT JSONL is 2,187,347 bytes. Every existing CSV/JSONL training source and rewrite replacement was audited individually. They must not be concatenated as independent corpora: the CSV/JSONL exports represent the same records, and transformations overlap their source.

Historical log shows a compact root replacement/placeholder repair followed by restoration of the original large dataset and incremental rewrite batches. Preserve this history. No tracked model checkpoints, adapters, prediction scores, DAPT corpus, DPO pairs, inference server, export script, GGUF converter, quantization validation or cloud provisioning implementation was found. Referenced model IDs were Qwen3 4B, 1.7B and 0.6B. The Colab/T4 comment and unused Unsloth environment variable were the only substantive training-environment remnants; they do not prove prior A100/H100 runs.

## Existing implementation and technical debt

| Script | Findings / disposition |
|---|---|
| `apply_rewrite_batches.py` | Working, auditable numbered rewrites; reused unchanged. All 815 matches apply. Temporary rebuild reproduced CSV, SFT JSONL and report byte-for-byte. |
| `prepare_firm_dataset.py` | Header discovery, metadata, audit flags, numeric-template cap and seed 42 exist. Random row splitting follows curation, allowing families across partitions; missing/duplicate rows disappear before audit. Retained as historical converter; new shared inspection and group splitter are the supported FIRM 3 path. |
| `curate_firm_dataset_v2.py` | Useful deterministic caps; exact-pair deduplication only. Can append eval-equivalent expert tasks without a leakage gate. Retained for history. |
| `merge_expert_jsonl_into_csv.py` | Supports in-place root replacement; example command overwrites root CSV. Not used this session. Writes lose original per-example provenance; multi-turn extraction keeps only last messages. |
| `build_firm_v3_large_dataset.py` | Recycles stock answers and a finite question/parameter library; adds `Variant N` to answers while prompts repeat. Pair deduplication misses repeated prompts. Do not use its default 1,000 synthetic rows to meet a size target. |
| `eval_firm_keywords.py` | No generation; lexical scoring only. Previously ignored core `expected_traits` and boundary `must_not_include`, and silently overwrote duplicate IDs. Those interface defects are fixed; scientific scoring remains separate. |
| `train_firm_qlora.py` | Existing provider-independent single-GPU QLoRA path reused. Previously no revision pin, seed flag, resume, manifests, loss masking policy or config reader; newer TRL changed `tokenizer` / `max_seq_length` APIs. Now has JSON profiles, dry-run preflight, hashes, completion-only loss, explicit resume and experiment metadata. GPU execution remains untested. |

The old `configs/qwen3_4b_qlora.yaml` points to absent `firm_v1_balanced_train.jsonl` and `...valid.jsonl`; the trainer did not consume YAML. It remains historical, not a runnable release profile. README mentioned a `data/raw` layout that had no tracked sources, and did not document any of the seven scripts with runnable commands. There were no README-named scripts that were missing; the problem was omitted commands and stale data paths. No `.gitignore`, tests, manifest generator, permanent hash contract or deterministic numerical scorer existed.

Original dependencies were unconstrained lower bounds, omitted explicit PyTorch, and combined data tooling with GPU training. New audit/split/eval tools need only the standard library. Separate data and legacy training profiles now pin direct dependencies. A fully resolved GPU environment lock/container digest still requires an actual compatible runner; pins are not a claim of GPU validation.

VPS observations: Python 3.13.7, 8 reported CPUs, about 31.3 GiB RAM; `nvidia-smi` is unavailable and PyTorch was not installed. No local model inference was attempted. An ignored `.venv-audit` contains tokenizer/audit dependencies, not model weights.

## Independently reproduced statistics

Counts below use whitespace-separated words (`\S+`), linear interpolated percentiles, and duplicate **extra records** beyond the first occurrence. All seven legacy CSV fields are complete. Both corpora have zero malformed or empty prompt/answer records.

| Metric | Original | Rewritten |
|---|---:|---:|
| Records | 2,532 | 2,532 |
| Exact duplicate pairs | 0 | 0 |
| Exact duplicate prompts | 0 | 1 |
| Exact duplicate responses | 114 | 53 |
| Mean prompt words | 14.58 | 15.05 |
| Mean response words | 21.02 | 33.30 |
| Response median / p95 / maximum words | 19 / 31 / 299 | 23 / 87 / 299 |
| Prompts participating in repeated numeric templates | 1,721 | 1,710 |
| Distinct repeated prompt-template families | 16 | 19 |
| Near prompt pairs at token Jaccard >= .85 | 387 | 377 |
| Near response pairs at token Jaccard >= .85 | 3,834 | 3,829 |

The README's counts of 2,532, zero missing fields and 114 duplicate responses reproduce. Its 16.4/23.4 word means, p95 32 and maximum 302 do **not** reproduce under the explicit whitespace method. An alternative `\w+` count gives original means 17.06/24.94 and maximum 313, also different. Treat the historical lengths as unsupported by an identified counting convention, not as current facts. Original response length ranges from 5 to 299 words; unusually short/long record locations are in the JSON snapshot.

Whitespace-normalized and Unicode/case-normalized exact pair counts stay zero. The rewritten duplicate prompt has two stylistically different answers giving the same ionization result, not a demonstrated contradictory numerical answer. More general contradictions require scientific review; normalized prompt/answer-variant candidates are reported without labeling them false.

Largest original prompt families: dopant ionization 576 plus a separately worded family of 98; NEP 180; D* 150; shot-limited SNR 140; shot current 120; Planck, photocurrent and Johnson families 100 each. The main ionization family remains 571 after rewriting. Most rows carry broad `Physics` labels (original 2,051/2,532), so categorical balance alone obscures mechanism coverage.

The activation-energy/temperature-only ionization recipes also need physical review: actual dopant occupancy depends on Fermi level, degeneracy and charge neutrality/carrier statistics. Correct substitution into a prescribed expression does not make it a universal detector-material ionization model. Remaining candidate rows are not approved merely because they passed selected arithmetic checks.

## Token audit

Measured with the **actual** `Qwen/Qwen3.5-9B` tokenizer at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Transformers 4.57.6 / tokenizers 0.22.2. Only tokenizer files were downloaded; no model weights. Saved tokenizer-file hashes and installed audit package versions accompany the snapshot. Chat counts use each existing system message where present; CSV fallback has user/assistant only, so those chat lengths are not directly comparable between formats.

| Metric | Original CSV | Rewritten SFT JSONL |
|---|---:|---:|
| Mean prompt tokens | 30.52 | 30.81 |
| Mean response tokens | 55.78 | 72.98 |
| Response p95 / maximum tokens | 82 / 754 | 132 / 754 |
| Maximum serialized chat tokens | 825 | 877 |

Expert batches average 149.16, 134.00 and 102.40 response words; the root expert file averages 78. No multi-turn conversations exist in these sources. This confirms more depth than the original seed, but does not establish correctness.

## Scientific quality and capability coverage

Heuristics count mentions, not validated reasoning. Original / rewritten counts respectively: equation-like notation 1,904 / 1,963; numerical-problem flags 1,569 / 1,578; derivation mentions 70 / 80; troubleshooting 2 / 51; empirical data mentions 52 / 443; uncertainty mentions 11 / 59. Unit-with-number flags are 1,784 / 1,771. These are not counts of correct calculations or genuine experimental datasets. The three citation-like flags in the original are code array indices, not verified literature citations. No externally sourced measurement arrays, image datasets or citation-backed expert corpus was identified.

Coverage in original / rewritten: HgCdTe or MCT 123 / 365; InSb 6 / 41; InGaAs 6 / 6; InAsSb 0 / 0; T2SL 0 / 0; MBE 0 / 2. LPE and MOCVD remain small. Hall inference, recombination, noise normalization, calibrated radiometry, contact/passivation effects and alternative hypotheses need richer, reviewed examples.

Seven exact legacy numerical families are recalculated with SI constants and a 2% tolerance for printed input rounding. Of 890 recognized examples, 736 pass this limited arithmetic check and 154 require review in **both** original and rewritten corpora: 150 D* unit errors, 3 Planck discrepancies and 1 Johnson discrepancy. The latter four may involve rounded inputs or numerical errors and are not automatically repaired.

The D* problem is systematic: `sqrt(A_m2 * bandwidth)/NEP` is in metres sqrt(Hz)/W. Labeling it Jones requires multiplication by 100. Example: area `1.370e-8 m^2`, bandwidth `85620.2 Hz`, NEP `1.069e-10 W` implies approximately `3.204e10 Jones`, not `3.204e8`. The audit and regression test expose this error deterministically.

Manual flags include dimensionally invalid conversion from A/W to V/W by a dimensionless divider; long-form Johnson arithmetic; unjustified parallel addition of RC/recombination rates; GR variance inferred from mean current without a PSD amplitude/population model; blackbody band radiances larger than total blackbody radiance; an apparent blackbody temperature hotter than both mixed scene components; and a truncated unreferenced HgCdTe bandgap polynomial. The latter is inconsistent with the full HSC relation described in the [NIST bandgap paper](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=14738). These are preserved and quarantined, not silently rewritten. `firm3_review_exclusions.json` records exact prompts, source hashes, locations and reasons. This is an AI audit; human expert approval is still pending.

## Evaluation contamination

All 30 existing eval records were inspected. Automatic checks cover exact prompts, Unicode/case/whitespace normalization, numerical-template equality, token Jaccard, character similarity and sufficiently long whole-answer equality. Nearest neighbors are recorded for manual review even below thresholds. No embedding-based semantic proof was attempted.

Confirmed original overlaps include exact `What does NASA do?` and latest-sports prompts, satellites differing only by punctuation, and a lasagna paraphrase. Rewritten SFT retains the sports exact match, satellites and lasagna equivalents, and a close cutoff/bandgap prompt corresponding to core 003. Root expert line 4 closely paraphrases core 004. Manual review found expert task equivalents for blackbody-to-voltage, transmission-slope diagnosis, passivation tradeoffs, Hall constraints, annealing with unchanged cutoff, IV insufficiency and rolloff de-embedding. These are explicit conservative exclusions; absence of an automatic flag is not proof of independence.

The original three files remain byte-for-byte unchanged and are frozen in `evals/firm_legacy_v1_manifest.json` with file/record hashes and public-rubric provenance. They measure legacy diagnostic behavior, not clean generalization. Foundational pretraining exposure remains unknown for any public benchmark.

An additional public development pilot has 8 linked idealized tasks and 22 structured numerical quantities. Its numerical oracles have CPU checks; it is versioned separately and requires expert review before becoming part of a permanent benchmark. It is not a claim of broad coverage, unseen test performance or actual experimental data. No model benchmark results were generated.

Pilot 007 deliberately checks the legacy absorption-thickness error; treat it as a regression task, not independently unseen evidence. The pilot as a whole is public development material. The permanent capability benchmark must be independently authored and reserved before corpus expansion.

## Infrastructure and resulting candidate data

Shared `firm_data.py` loads sources without silent filtering. `audit_firm_data.py` inventories every source, reports distributions/templates/near pairs, performs limited numerical checks and emits source-specific leakage candidates. `split_firm_data.py` deduplicates case-preserving exact pairs after trimming outer whitespace and connects numeric prompt/response templates and declared family/document IDs. Any automatic or reviewed exclusion quarantines the whole connected group. Caps apply within groups; seed 42 assigns whole groups to train/valid/test. This is deliberately conservative; false-positive quarantine can be resolved only with an explicit review trail.

Candidate inputs: rewritten SFT plus all four expert files, totaling 2,581 unique pairs. Output: **717 train / 65 valid / 106 test / 379 quarantine / 1,314 capped**. Generated rows remain local ignored artifacts; their hashes, source identities, exclusion manifest and reproduction command are tracked in `data/manifests/firm3_candidate_2026-10-05_v2_manifest.json`. These are **unreviewed candidates**, not an approved training release. Group allocation is not stratified and does not guarantee exact 80/10/10 record proportions. Related paraphrases below lexical thresholds can still span partitions; permanent test creation must use independently authored scenarios and declared families.

The active v2 splitter preserves unit/variable case and line structure during pair deduplication, while conservative normalized grouping/leakage flags remain separate. It also propagates every declared document/family link when merging duplicate pairs. An earlier local prototype is retained with an explicitly superseded manifest. Active output hashes reproduce under reversed source order, and eligible splits have zero candidates under the implemented lexical checks against all 38 eval prompts; this does not establish exhaustive semantic independence.

New numerical grading checks declared quantities, finite values, explicit supported unit conversions and relative/absolute tolerances. Physical correctness, equations, derivations, diagnosis and citations remain unscored human-rubric fields. Baseline generation uses an existing localhost server, sends prompts without gold values, retains raw responses/finish reasons and resumes by matching run metadata. Training preflight records revisions/hashes/seed/config, refuses mismatched partitions and hash changes, and supports resumable optimizer checkpoints. CI checks are read-only; their remote run status is separate from local verification.

## Reproduction and remaining debt

Commands are in README and handoff. Audit artifacts include source hashes, full category distributions, candidate locations and near-pair counts. Remaining: explicit dataset/source licensing, human review of flagged and unflagged examples, corrected versioned data release, a substantially broader independently authored benchmark, expert rubric calibration, GPU stack validation and contemporary VLM/hybrid model training integration. No historical trained FIRM checkpoint was recovered from this repository.
