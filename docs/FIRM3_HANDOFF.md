# FIRM 3 scientific-validation handoff

Updated 2026-10-06. Objective: trustworthy quantitative IR-detector adaptation, not plausible short answers. **E2 NO-GO.** Data/oracles and the native CPU integration path justify preparing a bounded E1B technical smoke; they do not yet justify substantive paid FIRM 3 fine-tuning. No paid compute was provisioned or launched.

## Repository / implementation / publication

- Git root `/home/User/FIRM`, origin `git@github.com:Kajin-0/FIRM.git` (Kajin-0/FIRM).
- Takeover independently verified clean/synchronized foundation branch at local/live `58b7e4db268730a10e42830407a2d7a0a1a246cd`; original 17 tests passed. No pre-existing changes needed preservation.
- Branch/upstream: `firm3/scientific-validation-e0` / `origin/firm3/scientific-validation-e0`; no merge to main.
- Implementation HEAD: **5050c5ce72a583d86f40cc9f767fc1ee91133233**, pushed and independently verified live, ahead/behind 0/0. Documentation/package-evidence commits follow it; obtain their final tip with `git rev-parse HEAD` (a document cannot contain its own commit SHA).
- Main remains `17351932a7337946d2017a5aa14c0a7201767096`; foundation remote remains 58b7e4d. Verify the final published tip with commands below.
- Implementation commits: e8d3634 scientific reviews/Jones repair; a3424f9 dev v1/oracles; 5d1bdab reviewed seed/dev v2; 4cfc88c independent data/runtime tests; 77024c6 pinned native profile/CPU gates; 69fd189 portable GPU/E0 protocols; 8af5243 compact pilot findings; 5050c5c completed E0 artifacts/reporting fixes.
- Original CSV, rewritten sources, expert corpora, legacy evals, pilot v1 and historical candidate v2 are preserved byte-identically. Published releases are immutable; regenerate to NEW paths. Local draft seed v1/v2, CPU fixture runs/venvs and package archives are preserved ignored artifacts.

## Scientific data state

- Pilot: all 8 cases/22 quantities independently rederived, no oracle error, no pilot v2. Explicit assumptions/units/conventions/tolerances in scientific-review doc and reference tests.
- Original E1 selection was the first 32 stable-ID-sorted training rows, not a seeded random sample: **25 accept / 1 correct / 6 exclude / 0 unresolved**. Five exclusions lack dopant carrier statistics/neutrality; one junction lacks necessary ni/degeneracy assumptions. One Poisson radiometry claim was bounded. Compact full row-review table is in FIRM3_SCIENTIFIC_REVIEW.md.
- Another 30 conceptual anchors individually accepted. AI analytic review, not human expert signoff; unknown source rights do not become known through correction.
- All **150 D* rows** confirmed individually: old answers approximately sqrt(A_m2*BW)/NEP_RMS but labeled Jones. Versioned `Jones-SI-area-v1` recomputes sqrt((A_m2*1e4)*BW)/NEP_RMS, with explicit units, old/new values/formula/parent/source hashes. Originals and unaffected families unchanged. Corrected rows are **all quarantined** as eval-equivalent, not leaked into training.
- `data/reviews/firm3_scientific_v1/firm3_scientific_review_v1.jsonl` records 2,889 entries (2,581 canonical plus flagged historical representations); manual decisions are separately source/hash-pinned. Four other arithmetic flags remain needs_review; subjective errors are not guessed away.
- Active `data/processed/firm3_reviewed_seed_v3/`: **45 train / 1 valid / 1 test / 159 quarantine / 0 capped / 725 excluded / 1,650 unreviewed**. Of 206 accepted/corrected rows, 159 are reserved (150 D* + nine reviewed numerical variants). The eligible 47 are conceptual smoke material; one-row heldouts are plumbing checks, not quality validation.
- Manifest SHA-256: **88fbf10096a8aaf040711d889ef94a48786347e34832de0d3dbaaf27f76a474a**. Seven JSONL outputs regenerate identically. Whole-family connected groups, cap3 and conservative quarantine retained. `firm3_reviewed_seed_v3_leakage.json` has zero implemented lexical/structural matches; semantic/base-pretraining exposure is unknown.

## Benchmark / actual E0

- Legacy30 and numeric pilot v1 remain frozen. Active **firm_science_dev_v2**: 40 cases,40 category labels,106 deterministic quantities; per-case assumptions/units/reference functions/tolerances/manual rubrics. Public AI-reviewed development, not unbiased hidden test. Dev v1 retained; v2 replaces only a legacy-related greybody scenario with a two-color Planck inverse.
- Dev v2 JSONL SHA **ebfd7d3e62231fb1710ccd572bab24a1233f06b0d570e964aa40d8054db68164**; manifest SHA **fec09c634d9bdde5993363dcf826229ef1208f223e3ab27236579c1f2810e692**.
- Actually run: installed Ollama0.32.15 `qwen3.5:9b`, Q4_K_M, CPU, artifact digest **6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7**. **HF upstream revision UNKNOWN.** No weights pulled and no CUDA/server provisioned.
- Primary local compact-v2 track: thinking off,temperature0,seed42,2048 output tokens/context8192,tool-free. Pilot **3/22 numeric,12/22 units,3/8 parse failures,zero truncation**. Development **9/106 numeric,65/106 units,16/40 parse failures,two truncations,39/40 terminal replies**. Case013 incomplete backend reply retained and fails; attempted coverage is not successful generation coverage.
- Legacy: each10/10 terminal; coverage proxies core0.307/boundary0.415/expert0.445, contaminated diagnostics only. Manual findings include passive-filter variance violation, absorption/reflection bound violation, wrong Jones and lock-in model, and correct structured values contradicted by prose.
- Raw outputs/run manifests/exact executed source snapshots, old verbose track, final `*_score_v2.json`, manual observations and `results_manifest_v1.json` are tracked under `evals/results/`. Valid unit aliases were corrected without changing numerical scores; old scores retained.
- Future runner now reports nonterminal generation errors and does not demand JSON from legacy prose. A live Bash wrapper edit caused exit2 after all inference sets completed; all IDs/hashes were verified. Do not edit running launch scripts. Manual findings are AI qualitative observations, not calibrated physical/citation scores.
- **Pinned HF E0 comparison: NOT RUN.** Primary standard track4096/8192/off; separate reasoning8192/16384/on or declared native-always-on. All models share protocol/budget within track; no gold/rubric values sent. Exact launch commands in FIRM3_E0_RESULTS.md.

## Modern profile / E1 gates / cloud

- Initial technical target **Qwen/Qwen3.5-9B**, pin **c202236235762e1c871ad0ccb60c8ee5ba337b9a**; not selected as a benchmark winner. Challengers in configs/firm3_e0_models_v1.json: Ministral14B Reasoning pin51f9210f3cd20f3452a80d5819d15dc61cc50630; Qwen3.8-27B pin1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; Qwen3-8B control pinb968826d9c46dd6066d109eabc6255188de91218. Authoritative licenses/architecture/revisions recorded; all these E0 runs are NOT RUN.
- Native **Qwen3_5ForConditionalGeneration + pinned AutoTokenizer**, text only/no pixels; preserve frozen full vision model. Exactly96 language MLP gate/up/down projections adapted. Full/DeltaNet attention, convolution/state, embeddings, LM head and visual modules frozen. BF16 rank4 alpha8,SDPA/native reference hybrid fallback. Modern NF4/FlashAttention/optimized kernels unvalidated; NF4 is refused.
- Actual random **58,208-parameter native CPU** fixture (3DeltaNet+1full attention) passed two TRL updates, prompt masks, finite/nonzero LoRA gradients, every frozen tensor unchanged, step1 checkpoint/resume to2, separate adapter reload with **zero full-logit error**. Real pinned tokenizer checked8 reviewed examples (90–276 tokens). Evidence `data/reviews/modern_profile_cpu_v2.json`; no9B/GPU quality claim.
- Pinned modern stack: Python3.13,torch2.9.1,Transformers5.18.0,PEFT0.21.2,TRL1.14.1,bitsandbytes0.50.2,Accelerate1.15.0,Datasets4.7.0; CPU pip-check passes. CUDA12.8 wheel/environment untested. Found/fixed TRL attention-mask column removal and CPU autocast/reload mismatch.
- **E1A GPU NOT RUN**: configs/firm3_e1a_reviewed.json, pinned Qwen3-.6B,32 reviewed train/1valid,10steps,NF4,r8/a16,2048tokens. Historical unreviewed32/65 config remains preserved and actual execution is refused.
- **E1B GPU NOT RUN**: configs/firm3_e1b_qwen35.json,8 reviewed train/1valid,2steps,512tokens,microbatch1,r4/a8,BF16/no quantization. Hard limited to8–16 rows/1–3steps. Measured full-size gradient/mask/frozen/VRAM/timing/checkpoint/resume/reload gates required before E2.
- Provider-neutral package/bootstrap/run/resume/reload/artifact collection ready; no credentials or provisioning code. Persistent output/cache required; checkpoint-boundary SIGTERM save; never overwrite outputs. Package-evidence section is appended after final archive validation.
- Planning only: A10040GB fit risk (estimated28–40GB class, not measured); **A10080GB preferred** first E1B; H10080GB alternative, speed unmeasured. Google shapes a2-highgpu-1g/a2-ultragpu-1g/a3-highgpu-1g; small A3 uses Spot/Flex-start. No invented cost; authorize a spending cap before rental.

## Reproduce and exact next experiment

```bash
cd /home/User/FIRM
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
python3 scripts/train_firm_qlora.py --config configs/firm3_e1a_reviewed.json --dry-run
python3 scripts/train_firm_qlora.py --config configs/firm3_e1b_qwen35.json --dry-run
bash -n scripts/bootstrap_firm_gpu.sh scripts/firm_gpu_run.sh scripts/run_firm_e0_suite.sh
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count HEAD...origin/firm3/scientific-validation-e0
git ls-remote origin refs/heads/firm3/scientific-validation-e0 refs/heads/main
```

**36 CPU tests PASS**, compile/bash syntax/preflights PASS, modern pip-check PASS; extracted package preflight/tests PASS. To regenerate data, use README review command with new directories. To repeat tiny native CPU proof, see GPU runbook/validator help in the existing isolated modern environment. Do not rerun the completed CPU baseline merely to resolve protocol bookkeeping.

Next training experiment is **bounded E1B only**, after explicit paid-GPU authorization/spending cap: extract verified package on approved A10080GB, `bash scripts/bootstrap_firm_gpu.sh E1B`; set persistent `FIRM_RUN_DIR`; `bash scripts/firm_gpu_run.sh E1B preflight`, then `train`, explicit interrupted `resume`, `reload`, `collect`, and stop rented GPU. Capture actual hardware/software/VRAM/timing/frozen/gradient/resume/reload evidence. Run exact-HF-revision E0 in the separate serving environment before domain adaptation.

E2 blockers: richer independently reviewed rights-cleared training examples (current47 inadequate), human scientific/semantic review and benchmark calibration, exact-revision E0, full-size CUDA profile/resume/reload/runtime measurements, explicit spending authorization. DAPT -> long-form SFT -> correctness/tools -> provenance RAG -> authentic multimodal -> validated quantization remains the roadmap. No mass literature download or E2 occurred.

Detailed evidence: [scientific review](FIRM3_SCIENTIFIC_REVIEW.md), [E0](FIRM3_E0_RESULTS.md), [GPU runbook](FIRM3_GPU_RUNBOOK.md), [spec](FIRM3_SPEC.md), [training plan](FIRM3_TRAINING_PLAN.md), [audit](FIRM3_AUDIT.md).
