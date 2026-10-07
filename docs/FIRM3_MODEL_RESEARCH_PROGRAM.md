# FIRM Model Research Program — Compact Infrared Specialist

**Status:** benchmark and governance infrastructure, not a newly trained specialist. **No cloud GPU instance authorized or created by this work.** This research plan supplements `FIRM3_SPECIALIST_CHARTER.md` and supersedes the earlier assumption that 9B Qwen3.5 automatically remains the final student.

## Objective function

**Maximize validated infrared/semiconductor science competence subject to total parameter count, actual resident weight memory, CPU/GPU inference latency, model rights, and practical Ollama deployment.** Do not optimize only synthetic formula accuracy or model branding.

Candidate size comparisons MUST use the **total weights needed at inference**, not just active/effective parameters or nominal label. A 4B-effective model with 8B total weights is not equal in memory cost to a 4B-total model.

## Independent evaluation bank

A private lockbox on the VPS lives at:

`~/.local/share/firm-private-bench/large_quant_v1/`

A private 256-bit seed lives at:

`~/.config/firm-private-bench/large_quant_seed` (permissions 0600).

The GitHub repository contains **benchmark generators and scoring logic only**, not any actual private evaluation prompts or gold labels. A small public scope development set is separately labeled and not a private release benchmark. These files MUST NEVER be added to a training set, published to GitHub, sent to candidate teachers as example problems during SFT, or used to optimize prompts iteratively.

| Tier | Families | Locked questions | Interpretation |
|---|---:|---:|---|
| Known formula, held-out inputs | 22 | 5,500 | Measures interpolation and numerical/units formatting on the same formula families used by synthetic training. **Not** evidence of scientific generalization. |
| Unseen formula families | 15 | 6,000 | Tests transfer across distinct quantitative equations absent from the existing 22-family synthetic SFT module; still template-generated and **not** independent expert conceptual review. |
| Scope / safety / model identity | manual groups | 46 | Curated public development rubric in `evals/firm_specialist_scope_v1.jsonl`; user-origin examples are **excluded from training**. Needs a separate genuinely private unseen scope holdout before product release. |
| Frozen science-dev v2 | 40 cases / 106 quantities | 40 | Previously measured Qwen3.5-9B exact baseline. Continue to keep gold away from future SFT. |
| Future expert-written release evaluation | TBD | target 400–1,000 | Needs external physics/fabrication review, process-grounded sources, reproducible solutions, adversarial incomplete-data questions, citations and uncertainty; **not created yet**. |

Quantitative checksum commitments:
- Known family SHA-256: `784c84569e48387a098b4788453fb6edef0a5f48d1039a6ce9ffe0666110ee6f`.
- Unseen family SHA-256: `b8a7c00539eba70874d7171287beaa062e4654aeb219e6c3fadf69cf51a4d943`.

Both partitions are deterministically regenerated with private seed, checked against analytical invariants and the existing synthetic SFT train/validation prompt set. Prompts explicitly request strict JSON so parse-compliance scoring is meaningful; units are supplied by the model. The generator refuses to overwrite a locked partition without an explicit force flag and updated checksum commitments. Their test prompts have never been given to a model as training examples by this benchmark-building workflow.

## Scientific assessment rubric

Numerical: schema compliance, physical unit, absolute/relative error, temperature assumptions and numerical magnitude; **joint numeric-plus-unit correctness** is primary. Separately flag refusal, truncation, backend errors and incomplete coverage. Score **each family separately** so easy units do not hide failure on hard radiation formulas.

Conceptual: independently reviewed reference answer with specific claims and traceable primary literature/textbooks. Mark incorrect atomic claims, fabricated evidence, overconfidence, unsupported process etchants, causal-mechanism inference from insufficient data and in-scope false refusals. Mark open/controversial topics as contested.

Scope: do not answer unrelated entertainment, consumer, politics, travel etc. However answer foundational EM, statistics, numerical methods, materials chemistry and scientific programming **when explicitly relevant to IR or semiconductor devices**. Emergency-safety cases can receive minimal urgent safety guidance.

Statistical: report denominators and 95% Wilson intervals at item level, plus bootstrap CIs by **family** for genuine domain-coverage claims. Never claim 11,500 independent scientific concepts: items within a formula family are correlated. Optimize on a public development set only; reserve private family-level holdout until final model selection.

Performance: measured loaded GPU RAM/CPU RAM, quantized model size, first-token latency, decoded tokens/s, context, output and reproducible launch settings. Grade no-thinking and budgeted-thinking tracks **separately**.

## Candidate roster (not ranked before measurement)

- `Qwen/Qwen3.5-4B`: official Apache-2.0 post-trained dense/hybrid 4B language model; native hybrid DeltaNet architecture; actual full checkpoint includes multimodal weights. Primary **compact student hypothesis**.
- `Qwen/Qwen3.5-4B-Base`: same nominal size, pretrained variant. Consider staged domain-adaptive pretraining + specialist SFT if reviewed source corpus is sufficiently diverse; do not assume this outperforms the post-trained checkpoint.
- `google/gemma-4-E2B-it`: **2.3B effective, ~5.1B total** including embeddings; Apache-2.0; candidate for aggressively small runtime if total memory checks out.
- `google/gemma-4-E4B-it`: **4.5B effective, ~8B total** including embeddings; Apache-2.0; candidate only if accuracy compensates for extra total memory.
- `google/gemma-3-4b-it`: 4B comparison, different license terms; review redistribution/training requirements before deployment.
- Existing `Qwen/Qwen3.5-9B`: engineering/control model. Previously measured exact E0 quantitative weaknesses mean it cannot be assumed to be a high-quality teacher.
- Existing local `firm:fast` is prompt-only Qwen2.5-3B-Instruct; existing `firm:baseline` is prompt-only Qwen3.5-9B. **Neither is the target trained model.**

Verify immutable commit SHA and effective license and import compatibility **before any training**, not just an unpinned model name. Source model pages:
https://huggingface.co/Qwen/Qwen3.5-4B
https://huggingface.co/Qwen/Qwen3.5-4B-Base
https://huggingface.co/google/gemma-4-E2B-it
https://huggingface.co/google/gemma-4-E4B-it
https://ai.google.dev/gemma/docs/core/model_card_4

## Dataset development and evaluation stages

1. Source audit: establish textbook/article license, version, DOI and provenance for each factual claim. Avoid scraping copyrighted literature into training indiscriminately. Initial metadata-only source-candidate registry: `data/firm_scientific_source_candidates_v1.jsonl` (7 papers, all **training-blocked** until rights verification and independent review).
2. Concepts and causal explanations: materials identity, narrow-gap semiconductor devices, epitaxy, LPE/MBE, interfaces, wet etching, electronics, optical readout, cryogenic physics and thermal radiometry.
3. Numerical curriculum: original parameterized equations with independent oracles and explicit SI conversions, supplemented by reasoning-transfer cases **from DIFFERENT formula families**.
4. Specialist judgment: evaluate when measurements are insufficient, when uncertainty matters, and whether a claimed failure mechanism is diagnosable.
5. Selective answering: positive examples in adjacent, relevant physics; negative examples outside scope (not copied from the heldout stress prompts); safety exception.
6. Full scientific benchmark + refusal benchmark on selected compact checkpoint; rank only by measured Pareto frontier of accuracy vs real memory/latency.
7. Train progressively and compare exact weights at each stage; checkpoint and validate scientific non-regression. Never deploy merely because training loss declined.
8. Convert trained weights to Ollama only after exact-base adapter merge and a matching scored roundtrip. Model identity and revision must be accurately exposed.
9. Promote a candidate only after withheld expert release set and user acceptance criteria pass.

## Training cost/security

**No unrestricted Google GPU.** Existing experimental GCP project is currently credential-revoked. Any future provisioning requires explicit confirmation of remaining promotional credits (not a nominal $300), a project-wide resource inventory, one-GPU regional quota, time-bounded VM termination with auto-DELETE, deterministic checkpoint evacuation, and verification of no disks/VMs/IPs/commitments after deletion. A billing budget is an alert, never a hard stop. Build data, benchmark generators, review tools and CPU tests on VPS without cloud expense.

### Usage

Private generator (seed is auto-created once, locally, with 0600 permissions):

```bash
cd /home/User/FIRM
python3 scripts/build_firm_large_benchmark.py
```

When inference outputs are ready (each JSONL line `{"id":"...","response":"{\"quantities\":...}","finish_reason":"stop"}`), score a complete tier:

```bash
python3 scripts/score_firm_large_quant.py --tier unseen_family_transfer \
  --predictions /path/to/predictions.jsonl --output /path/to/score.json
```

Scorer rejects duplicate IDs, unknown IDs, missing gold, incomplete predicted sets (unless explicit diagnostic `--allow-partial`), invalid JSON and wrong units. It emits only aggregates/per-family scores, **never gold values**.
