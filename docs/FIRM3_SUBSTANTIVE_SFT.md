# FIRM quantitative model training — substantive experiment

**Status (2026-10-06): dataset and training/evaluation recipes prepared and CPU-validated; new substantive adapter weights are NOT trained yet.** This is not E1B. No cloud/GPU charges were incurred while preparing this curriculum.

> **Scope update (2026-10-06):** This numeric-only 9B recipe is an engineering module, **not the final FIRM specialist training strategy**. The current authoritative product target is a compact (~3–4B, model selection pending) infrared-photonics/semiconductor specialist with native in-domain competence and learned abstention from unrelated questions. See `docs/FIRM3_SPECIALIST_CHARTER.md` and the strictly held-out `evals/firm_specialist_scope_v1.jsonl`. Do not launch the 660-step recipe and call it a complete specialist: first assemble the reviewed subject-matter curriculum and separately sourced out-of-domain refusal set, and benchmark the candidate student checkpoints. User stress prompts stay held out.

## Deliverables already present

- Original, programmatically checked corpus at `data/processed/firm3_synthetic_quant_v1/`: **2,640 train / 264 validation** across **22** quantitative IR-detector physics families.
- Generator and independent analytical invariants: `scripts/build_firm_quantitative_sft.py`.
- Training entry point: `scripts/train_firm_quant_sft.py`.
- Before/after BF16 comparison: `scripts/eval_firm_quant_sft.py`.
- Local **untrained** Ollama profile `firm:baseline` using the installed Qwen3.5-9B GGUF, and responsive untrained `firm:fast` using existing Qwen3.5-0.8B-Text Q5 weights.
- Earlier two-step E1B adapter archive, which is **not** a substantively fine-tuned model.

**Scope warning:** This synthetic dataset teaches unit-aware calculations and structured outputs, not literature synthesis, correct diagnosis of unknown real detectors, independent research, or scientific citation. It is programmatically formula-verified, **not independently expert reviewed**. All validation examples use the same 22 formula families and different input values. No E0 gold labels or source solutions were imported. Preserve untouched external E0.

## Experimental objective and grading

Hypothesis: one-pass BF16 LoRA SFT on a formula-verified corpus improves strict JSON compliance, physical units and numerical calculations on held-out parameters without regressing the earlier frozen public IR evaluations. An explicit **paired BF16 evaluation** on the unchanged pinned base and post-SFT adapter is required before claiming improvement.

The default substantive recipe uses:
- Qwen/Qwen3.5-9B, immutable revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Native Qwen3.5 conditional-generation BF16 model, single GPU, language MLP gate/up/down adapters only, 96 intended targets; attention/DeltaNet, head, embeddings and vision frozen.
- LoRA r=8, alpha=16, no quantization, no dropout, microbatch 1, gradient accumulation 4, 660 optimizer steps, maximum sequence 768, learning rate 3e-5, cosine scheduling and 3% warmup.
- Save every 110 steps, validate every 220 steps, retain 3 latest checkpoints, guard every step against trainable frozen gradients, report actual VRAM and train loss.

The L4 2-step E1B fit does **not guarantee** that this longer/stronger adaptation fits; a real CUDA evaluation is still required. Use provider-enforced runtime auto-deletion and **copy checkpoints off the temporary VM regularly**. A warning-only cloud budget is never a hard billing limit.

## On an already-authorized, bounded GPU VM

Extract the hash-verified package from the VPS, using the original package structure. Pin Python 3.13 and install CUDA 12.8 Torch and `requirements-modern-gpu.txt` (not `requirements-data.txt`, which has a conflicting legacy pandas pin). Example, from the package root:

```bash
python3.13 -m venv .venv-quant
.venv-quant/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv-quant/bin/python -m pip install -r requirements-modern-gpu.txt
.venv-quant/bin/python -m pip check

.venv-quant/bin/python scripts/train_firm_quant_sft.py --out /persistent/FIRM/quant-sft-v1 --dry-run
.venv-quant/bin/python scripts/eval_firm_quant_sft.py --out /persistent/FIRM/quant-e0 --dry-run
```

Run the baseline evaluation **before** adaptation on the exact HF revision, then train, then evaluate the adapter:

```bash
export HF_HOME=/persistent/hf-cache
.venv-quant/bin/python scripts/eval_firm_quant_sft.py --out /persistent/FIRM/quant-e0
.venv-quant/bin/python scripts/train_firm_quant_sft.py --out /persistent/FIRM/quant-sft-v1
.venv-quant/bin/python scripts/eval_firm_quant_sft.py --adapter /persistent/FIRM/quant-sft-v1 --out /persistent/FIRM/quant-adapted
```

Never start another run in an existing output directory. For deliberate resume, pass `--resume-from-checkpoint /persistent/FIRM/quant-sft-v1/checkpoint-N` with the same source revision, dataset/manifest and configured budget. Preserve score JSON, all raw generations, model metadata, adapter, checkpoint state and exact software versions. Stop and delete the temporary GPU VM and disk promptly after transferring artifacts.

## Test locally through Ollama

Run the already-created untrained baseline models **now** from the VPS:

```bash
ollama run firm:fast
ollama run firm:baseline
```

The 9B model on the CPU-only VPS may start/respond very slowly. `firm:fast` is more responsive, but smaller and less reliable. Neither contains the new curriculum-trained weights.

**After real adapter training:** importing a Qwen3.5 LoRA onto an arbitrary preexisting GGUF is **not validated** and risks a base-revision mismatch. The intended release procedure is:

1. Reconstruct the exact pinned Hugging Face base and load the saved PEFT adapter.
2. Validate adapter reload, then merge into the exact base using PEFT on appropriately sized hardware.
3. Export the merged model to GGUF using a llama.cpp converter verified for this Qwen3.5 hybrid architecture, quantize from high precision, and run a separate numerical regression gate.
4. Import matching GGUF into Ollama via `FROM /path/to/model.gguf` and `ollama create firm:quant-v1 -f Modelfile`.
5. Only call the output `firm:quant-v1` when the trained weights are demonstrably loaded and the evaluation passes.

Ollama supports GGUF import, but Qwen3.5-specific LoRA conversion has had upstream compatibility bugs. Never report success without a conversion/load/score roundtrip.

## Stop/go

**Go:** measurable paired improvement in held-out strict JSON, numeric and unit correctness, no significant regression on frozen E0 material and physically defensible sample answers.

**No-go:** high parse failures, hallucinated citations, numerical/units regression, invalid training provenance, unreviewed rights-sensitive source material, or inability to verify the exact base and merged weights. Synthesized results alone are not a production research model.
