# FIRM Compact Student Experiment — Qwen3.5-4B

**Research status:** CPU candidate screening complete; initial specialist SFT corpus assembled; exact pinned Hugging Face 4B GPU train/evaluation **not run**. No new learned weights or Ollama-trained FIRM release exists. **Do not call prompt-only `firm:fast` a fine-tuned FIRM release.**

## Candidate and baselines

- Provisional primary student: `Qwen/Qwen3.5-4B`, Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0. The exact native configuration has 32 language layers, hidden width 2560, and 96 language MLP gate/up/down projection targets. Native hybrid language/vision profile; no NF4 path is approved.
- Alternative compact control: Ollama `qwen2.5:3b-instruct` (unrelated architecture/revision, instruct model). Exact Hugging Face pin not yet reconstructed for a valid same-precision comparison.
- Previous large-model control: pinned `Qwen/Qwen3.5-9B`, poor measured physics numeric E0; not a trustworthy automated fact-labeling teacher.

## Initial CPU-only screening (10 prompts, same instructions, thinking off)

| Model | Ollama model size | Mean wall seconds / request | Requests | Observed findings |
|---|---:|---:|---:|---|
| qwen3.5:4b | 3.3 GB | 11.15 | 10 | InSb identity/gap correct; cutoff photon energy ~0.153 eV correct; two explicit out-of-domain refusals; minority-carrier diffusion equation reasonable. Serious contradiction about CZT substrate; weak alternative-substrate claims; identity falsely says specialized weights already trained; some answers truncated at 230 tokens. |
| qwen2.5:3b-instruct | 1.9 GB | 4.91 | 10 | Faster, but InSb gap incorrectly ~0.8 eV; extremely wrong wavelength-to-gap conversion; incorrect D* formula; wrong typical HgCdTe substrates; answered an out-of-scope cooking question; problematic minority-diffusion derivation. |

Raw responses: `runs/compact-cpu-screen/qwen35-4b.jsonl` and `runs/compact-cpu-screen/qwen25-3b-instruct.jsonl` (local developer records, not yet an official quantitative model ranking).

**Caveats:** CPU-only, very small manually inspected development sample; different model training/quantization and different underlying revisions, fixed 230-token response cap, no exact-HF revision compatibility demonstrated. No statistically meaningful claim of superiority or production readiness follows from this screen. Specialized system prompt may explain some scope refusals without learned specialization.

## Prepared concept/scoping corpus

`data/processed/firm3_specialist_concepts_v1/` contains **657 train / 101 validation** original-editorial examples from **51 infrared/semiconductor concept cards and 50 unrelated-topic scope cards**. Validation reuses concepts with distinct phrasing, therefore measures paraphrase retention rather than true out-of-distribution domain generalization. Each card starts with **training_eligible=false** and a pending review entry in `data/reviews/firm3_specialist_concepts_v1_review.csv`.

`data/processed/firm3_specialist_4b_mix_v1/` contains an explicitly **provisional research-only** weighted mix:

| Split | Oracle-verified original numeric | Conceptual (reweighted) | Scope refusal (reweighted) | Total |
|---|---:|---:|---:|---:|
| Train | 2,640 | 1,071 | 600 | **4,311** |
| Validation | 264 | 51 | 50 | **365** |

Original numerical questions are programmatically verified. Conceptual statements and scope classifications are **not independently expert approved**. The candidate research mix is explicitly labeled `PROVISIONAL_UNREVIEWED_RESEARCH_ONLY`; production use is blocked pending the card review ledger. This is a **research-only** training attempt, not a proposed released model. No benchmark prompts, original user transcripts, or private 11,500-question bank were imported into the mix; guardrails enforce review status and locked hashes. Never conflate 4,311 rows with 4,311 distinct independently expert-validated concepts.

## 4B training specification (GPU candidate, not yet validated)

`scripts/train_firm_specialist_4b.py`:
- Exact pinned 4B HF model, native BF16 and SDPA; LoRA on 32 × 3 language MLP projections only. Vision, head, attention/DeltaNet and embeddings frozen.
- Rank 8, alpha 16, no dropout, microbatch 1 × gradient accumulation 4, 1024 tokens maximum, 1,100 optimizer steps, learning rate 3e-5, cosine decay with 3% warmup, periodic checkpoint/validation.
- Train/validation identity, dataset hashes, review ledger, model revision and vocabulary boundaries checked before GPU work. The preview is tested on CPU; exact 4B GPU training compatibility remains unmeasured.
- **Default refuses to train** until all 101 concept/scope cards are independently approved; `--allow-provisional-data` must be explicitly requested for experimental nonproduction research.

When provisionally evaluating new weights, benchmark both the unchanged pinned base and adapted model on identical settings; compare numerical correctness + units on the 11,500 private quantitative problems, the frozen E0 science suite, selected novel expert reasoning problems, in-domain acceptance, outside-domain refusals, hallucination rates, model identity accuracy, output lengths, CPU RAM/latency and quantized Ollama roundtrip. Never train on private gold.

## Reproducibility and preflight, no GPU

```bash
cd /home/User/FIRM
python3 scripts/assemble_firm_specialist_dataset.py --dry-run
python3 scripts/train_firm_specialist_4b.py --allow-provisional-data \
   --data data/processed/firm3_specialist_4b_mix_v1 \
   --out runs/firm-specialist-4b-experiment --dry-run
python3 -m unittest discover -s tests -q
```

No Google GPU may be provisioned until promotional credit and remaining headroom are verified (a nominal $300 entitlement is NOT proof of balance), auto-delete and secondary shutdown safeguards are specified, checkpoint evacuation is tested and the user has explicitly accepted the spend risk. Budgets are alerts rather than hard caps.

## Training-source rights

Do not assume an online textbook permits LLM training. As of October 2026, OpenStax **University Physics Volume 3** explicitly states its materials may not be used for LLM training without prior written permission, notwithstanding its separate Creative Commons reuse terms: https://openstax.org/books/university-physics-volume-3/pages/1-summary. Do not ingest it. Use NIST CODATA data as a reference for physical constants, subject to source/terms verification: https://www.nist.gov/programs-projects/codata-values-fundamental-physical-constants.

## Promotion criteria

Pretrained 4B, 9B, and other eligible contenders must be evaluated by independent physics concepts as well as formula templates. Promote only when numerical+unit correctness, in-scope answer rate, scientific claim grounding, evidence calibration and appropriately refused unrelated prompts outperform the base without excessive inference latency or memory. Distinguish an exploratory first LoRA from a scientifically approved released model.
