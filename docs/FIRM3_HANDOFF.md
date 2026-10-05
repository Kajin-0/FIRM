# FIRM 3 handoff

Updated 2026-10-05. Objective: build a quantitatively reliable IR-detector research model through reviewed data, controlled adaptation and evidence-based evaluation. This session established the foundation; it did not produce trained weights or model scores.

## Repository and Git

- VPS checkout `/home/User/FIRM` is GitHub `Kajin-0/FIRM`, origin `git@github.com:Kajin-0/FIRM.git`.
- Takeover: clean `main`, local/live remote HEAD both `17351932a7337946d2017a5aa14c0a7201767096`; ahead/behind `0/0` after safe fetch.
- Working branch: `firm3/audit-and-foundation`, upstream `origin/firm3/audit-and-foundation`. No merge to main.
- Implementation HEAD: `bbd30af60fd11f689311fd6f3402f2bcf01a198b`, successfully pushed. The documentation commit containing this handoff follows it; obtain the final tip with `git rev-parse HEAD`. Its own SHA cannot be embedded in its contents.
- Implementation commits: `146b0c0` audit/quarantine; `e2309c5` numerical pilot/baseline; `bbd30af` reproducible smoke training. Final documentation is a separate commit.
- Remote publication was verified at the implementation HEAD before this document was committed. Verify the latest documentation tip using `git rev-list --left-right --count HEAD...origin/firm3/audit-and-foundation` and `git ls-remote origin refs/heads/firm3/audit-and-foundation`.
- Original CSV, curation transformations, processed sources and three legacy eval files were preserved byte-for-byte. No original uncommitted work existed. Never reset/force-push or overwrite an existing generated release.

## Findings and data state

- Original CSV: 2,532 complete examples, zero exact duplicate pairs/prompts, 114 repeated response extras. Whitespace-word means: input 14.58, output 21.02; response p95 31, maximum 299. Historical length claims do not reproduce.
- Rewritten corpus: 2,532 examples, mean response 33.30 words. Four expert sources add 49 examples. Numbered rewrite batches are transformations, not additional independent training examples.
- Numeric families dominate: 1,721 original prompts participate in repeated templates. All 150 D* calculation answers miss the factor 100 needed for Jones when area is in square metres. Long-form sources contain further unit/arithmetic/measurement-model errors. Dopant-ionization recipes lack necessary statistics/neutrality assumptions. See review exclusions and audit before training.
- Latest local candidate directory: `data/processed/firm3_candidate_2026-10-05_v2/`; counts **717 train / 65 valid / 106 test / 379 quarantine / 1,314 capped**. All 2,581 inputs are accounted for. These candidates remain scientifically unreviewed, with unknown source licensing.
- Generated candidate rows are ignored in Git. Tracked snapshot: `data/manifests/firm3_candidate_2026-10-05_v2_manifest.json`; active training manifest is inside the local directory. Reproduce into a NEW directory with the README command. Row hashes reproduce; the regenerated manifest records the current Git SHA.
- The earlier candidate snapshot is retained and explicitly superseded. Use v2: exact-pair identity preserves unit case (mW and MW must not collapse); duplicate origins/document families stay linked.
- Audit snapshot `data/audits/firm3_2026-10-05/` covers all 36 existing source files. Actual pinned Qwen3.5-9B tokenizer counts, scientific checks, duplicate distributions, leakage candidates, takeover inventory, byte-identical rewrite reproduction and v2 reversed-input split verification are recorded there.

## Benchmark state

- Three legacy eval files, 30 cases, preserved and hashed in `evals/firm_legacy_v1_manifest.json`. Training contains exact/paraphrased/equivalent tasks; legacy results are diagnostic coverage only.
- Separate `evals/firm_numeric_pilot_v1.jsonl`: 8 idealized problems, 22 deterministic numerical quantities, explicit units/tolerances and manual physical/diagnosis rubrics. Public answers; expert review pending. Case 007 is an explicit legacy-error regression, not independent unseen evidence.
- New candidate train/valid/test partitions have zero lexical/structural leakage candidates against these 38 evals under the implemented detector, plus explicit equivalent-task exclusions. This does not establish absence of semantic leakage.
- A permanent benchmark still needs independently authored, expert-reviewed, document/family-disjoint scenarios and controlled solution access. Preserve published releases; correct through new versions.
- Numerical grading is deterministic with an explicit unit-conversion whitelist. Missing/invalid outputs fail. Subjective physics, equation, derivation, diagnosis and citation metrics remain manual; no perfect automatic judge is claimed.

## Implemented paths and model strategy

- Shared stdlib data loader; expanded audit; conservative grouped splitter/quarantine; frozen manifests; numerical schema/oracles/grader; resumable localhost-compatible baseline runner; corrected legacy keyword coverage handling.
- Existing QLoRA trainer improved rather than replaced: immutable model revisions, hashed inputs, config validation, lazy GPU imports, completion-only training, no silent truncation, seed/run metadata and checkpoint resume. CPU tests and preflights pass; actual GPU masks/gradients/resume/reload remain unverified.
- Minimal experiment schema and separate pinned data/training requirements; read-only CPU CI. VPS: Python 3.13.7, 8 CPUs, about 31 GiB RAM, no accessible CUDA GPU. Ignored `.venv-audit` contains only audit/tokenizer dependencies, not model weights.
- Provisional E0 choices: **Qwen3.5-9B** lightweight, **Gemma 4 12B / Ministral 3 14B Reasoning** medium challengers, **Qwen3.8-27B** research scale. Qwen3-8B is a conventional-stack control; Qwen3-0.6B is smoke-only. Publisher evidence/licenses/revisions are recorded in the training plan and model manifest; no FIRM score exists.
- Modern hybrid/vision candidates are deliberately rejected by the pinned legacy causal trainer. Validate a separate native loader, processor, PEFT targets and training profile before adapting them.
- Architecture: rights-reviewed DAPT -> expert/long-form scientific SFT -> reviewed correctness -> scientific tools -> provenance-preserving RAG -> authentic multimodal analysis -> validated quantized deployment. The spec distinguishes parametric physics from retrieved paper-specific claims.
- Cloud plan: 9B QLoRA initially one A100 40 GB; medium/27B work one A100/H100 80 GB subject to measured fit. DAPT/full updates may need sharding. VRAM ranges and verified advertised hourly prices are planning estimates, not measured performance or a spending authorization.

## Checks and exact next steps

From `/home/User/FIRM`:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/train_firm_qlora.py --config configs/firm3_e1_smoke.json --dry-run
python3 scripts/run_firm_baseline.py --eval evals/firm_numeric_pilot_v1.jsonl --model Qwen/Qwen3.5-9B --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --out predictions/E0-qwen35-9b.jsonl --dry-run
python3 scripts/audit_firm_data.py --out data/audits/new_snapshot --date 2026-10-05
git status --short --branch
git rev-parse HEAD
```

Seventeen CPU tests passed, including independent radiometric/PSD integration, Jones conversion, leakage/group determinism, malformed inputs, frozen hashes, no-gold baseline requests, interrupted mock resume and training preflight. Actual token audits and legacy rewrite byte reproduction passed. See README for optional offline tokenizer audit, candidate regeneration and scoring commands. Mock predictions are temporary test fixtures; no benchmark results were fabricated.

Immediate next work: expert-review the 8 pilot oracles/rubrics and 32 selected smoke rows; create versioned corrections for flagged training data and resolve provenance/rights. Run **E0** on an already-running compatible server at the pinned Qwen3.5-9B revision, removing only baseline `--dry-run`; grade with `eval_firm_science.py` and manually assess reasoning. Do not provision a paid endpoint merely for this check.

Exact next **training** experiment: **E1**, `configs/firm3_e1_smoke.json`, pinned Qwen3-0.6B, 32 training/65 validation rows, seed 42, NF4, rank 8/alpha 16, sequence 2,048, microbatch 1, accumulation 4, **10 optimizer steps**, save/eval every 2. After subset inspection and compatible GPU/environment validation, remove trainer `--dry-run`; inspect completion labels, test stop/resume and reload the adapter. No domain-quality conclusion from E1. E2 waits for reviewed data and a validated contemporary model profile.

Outstanding: human scientific and semantic-leakage review; source/data licenses; permanent benchmark; GPU environment resolver/forward/backward/resume validation; actual E0 scores; native modern-model training integration. No mass literature download, GPU job, cloud spending, checkpoint export or model availability claim based solely on memory occurred.

Detailed evidence: [audit](FIRM3_AUDIT.md), [specification](FIRM3_SPEC.md), [training plan](FIRM3_TRAINING_PLAN.md), [reproduction commands](../README.md).
