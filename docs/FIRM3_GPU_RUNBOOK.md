# FIRM 3 GPU execution gate

No GPU was rented, billed or trained in this session. E1A/E1B full-size CUDA execution is **NOT RUN**. E2 is **NO-GO**. This runbook prepares a bounded technical smoke on an already authorized machine; it is not authorization to provision one.

## Native candidate profile and evidence

The first technical candidate is `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, selected for the initial integration test, not declared an E0 winner. The pinned checkpoint/config/index and library source were inspected. It is a vision-language checkpoint with 32 language layers in repeating three DeltaNet/one full-attention blocks. Use **`Qwen3_5ForConditionalGeneration`**, not the legacy causal loader. References: [publisher model card](https://huggingface.co/Qwen/Qwen3.5-9B), [native Transformers implementation/documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

For text-only data, use the pinned **AutoTokenizer**, explicit non-thinking chat template, and native full model with no pixel inputs. AutoProcessor is needed when image/video inputs are introduced; this profile rejects visual placeholders and preserves the frozen vision module. TRL receives already-tokenized text and a completion collator, so it does not invoke image preparation. No embedding/tokenizer resizing, MTP generation, vision adaptation, packing or custom remote code is enabled.

LoRA targets exactly **96 language MLP projections**: `model.language_model.layers.N.mlp.{gate_proj,up_proj,down_proj}`. Vision, embeddings, LM head, norms, full-attention projections and DeltaNet projections/convolution/state parameters stay frozen. Initial BF16 LoRA deliberately leaves hybrid attention adaptation for a subsequent measured ablation. The native class ignores optional `mtp.*` checkpoint tensors; any other unexpected/missing/mismatched loading keys abort.

Initial attention: SDPA for full attention; native reference PyTorch convolution/DeltaNet fallbacks, with no unpinned kernel download. Transformers exposes FlashAttention support, but hybrid optimized kernels/FlashAttention are **not validated** here. Reference fallback costs can be material at long sequence lengths. Gradient checkpointing uses non-reentrant mode; adapter gradient and frozen-parameter checks run before every optimizer update.

The separate legacy profile retains AutoModelForCausalLM + NF4 bitsandbytes + PEFT preparation. Generic 4-bit support does not prove Qwen3.5 hybrid-state/vision exclusion correctness. **Modern NF4/bitsandbytes execution is unvalidated and rejected by the current profile.** BF16 is the first gate. See [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

Actual CPU architecture evidence is in `data/reviews/modern_profile_cpu_v1/`; the final code rerun with tool/tokenizer hashes and measured artifact sizes is `data/reviews/modern_profile_cpu_v2.json`: 58,208-parameter random native fixture, three DeltaNet layers plus one full-attention layer, 12 MLP targets, two finite-gradient updates, all frozen tensors exactly unchanged, save at step 1/resume to step 2, adapter reload with **zero** full-logit difference. Real pinned tokenizer masks were checked on the eight E1B rows: 90–276 total tokens, nonempty assistant targets and masked prompts. This is software evidence; it does not establish 9B GPU/BF16/NF4 compatibility or domain performance.

Two observed fixes are explicit: preserve attention_mask with `remove_unused_columns=False`; disable TRL's default CPU BF16 for the FP32 fixture and use the same autocast context for actual GPU probe/reload. Adapter reload is a separate process with the same pinned base weights and numerical tolerance, not a claim based on successful saving alone.

## Pinned environments and experiments

| Profile | Environment | Data / optimizer steps | Precision / adaptation |
|---|---|---|---|
| E1A reviewed successor | Python 3.11; existing requirements-training.txt | 32 reviewed train, **1 reviewed valid**; 10 steps; sequence 2048; microbatch 1, accumulation 4 | Qwen3-0.6B pinned; NF4; rank 8, alpha 16 |
| E1B native candidate | Python 3.13; requirements-modern.txt/common lock | 8 reviewed train, 1 valid; **2 steps**; sequence 512; microbatch 1, accumulation 1 | Qwen3.5-9B pinned; BF16 base; rank 4, alpha 8; AdamW |

The original 32/65-row E1 config remains historical and unchanged; its unreviewed rows are refused by actual training. The reviewed E1A uses one validation row for plumbing. Its loss is not a quality metric. E1B is hard-limited to 8–16 rows and 1–3 steps; raising its budget requires an explicit new validated profile and passing E2 gates.

Modern CPU-tested versions: torch 2.9.1+cpu, Transformers 5.18.0, PEFT 0.21.2, TRL 1.14.1, Accelerate 1.15.0, Datasets 4.7.0, bitsandbytes 0.50.2. GPU bootstrap selects torch 2.9.1 CUDA 12.8; its NVIDIA wheel dependencies and GPU compatibility remain unmeasured. CPU common dependency versions are pinned, and every GPU run captures `pip freeze`. Bootstrap requires the indicated Python version and one visible CUDA GPU; it does not install drivers, create credentials or provision instances.

## Hardware planning — estimates, not measurements

| GPU | E1A | E1B initial BF16 / 512 tokens | Next decision |
|---|---|---|---|
| A100 40 GB | Expected ample room for 0.6B NF4 | Estimated roughly 28–40 GB memory class; reference DeltaNet overhead may exceed the budget | Try only with an agreed spending cap; stop on OOM, record it, preserve checkpoint; do not automatically resize |
| A100 80 GB | Unnecessary capacity for this small smoke | Preferred first modern gate for headroom | Measure actual allocated/reserved peaks and step time before choosing E2 hardware |
| H100 80 GB | Unnecessary capacity unless already available | Same memory class; speed advantage is unmeasured | Compare measured cost/throughput after the initial gate |

About 19–20 GB of BF16 full-checkpoint weights is only a lower bound; activations, mixed-precision state, optimizer, allocator and reference kernels add overhead. Rank-4 language MLP adapters add about 6.3M parameters. No actual GPU memory or throughput number is reported. Multi-GPU is unnecessary for E1B and refused by this profile. Do not use these smoke estimates to price DAPT/full tuning.

Google Compute Engine examples are `a2-highgpu-1g` (A100 40 GB), `a2-ultragpu-1g` (A100 80 GB), `a3-highgpu-1g` (H100 80 GB). Google's small A3 High shapes require Spot or Flex-start; availability/quota/region remain external gates. These are verified shape examples, not claims about the user's earlier Google service or the cheapest provider. Check current regional GPU+CPU+disk prices and preemption policy before approval; no dollar quote or spending limit is invented. [Official machine families](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines), [official GPU pricing](https://cloud.google.com/compute/gpus-pricing).

## Provider-neutral package and exact commands

On this VPS, after committing the implementation:

```bash
python3 scripts/package_firm_gpu.py --out runs/packages/firm3-e1b-ready.tar.gz
```

The archive includes committed code/configs, pinned requirements, reviewed dataset and lineage, preserved evals/manifests, tests and runbooks. It contains no model weights, virtualenv, Git internals, .env or credentials. An embedded manifest verifies every payload hash on preflight. Extract into a **new empty directory** on an already approved GPU machine. Alternatively clone this branch at the recorded commit. Mount a persistent disk for outputs and Hugging Face cache; never rely on a preemptible boot disk alone.

```bash
bash scripts/bootstrap_firm_gpu.sh E1B
export FIRM_RUN_DIR=/persistent/FIRM/E1B-qwen35-bf16-seed42
bash scripts/firm_gpu_run.sh E1B preflight
bash scripts/firm_gpu_run.sh E1B train
bash scripts/firm_gpu_run.sh E1B reload
bash scripts/firm_gpu_run.sh E1B collect
```

Use E1A instead of E1B for the legacy smoke. Bootstrap prepares software only. Actual train downloads the pinned base weights and runs the bounded config. Preflight loads no weights. Output work is never overwritten. If a run is interrupted after checkpoint 1:

```bash
bash scripts/firm_gpu_run.sh E1B resume "$FIRM_RUN_DIR/checkpoint-1"
bash scripts/firm_gpu_run.sh E1B reload
```

Checkpoints retain optimizer, scheduler, RNG and trainer state, with two checkpoints kept. SIGTERM requests save/stop at the next optimizer boundary; an abrupt eviction before a boundary may require restarting in a new run directory. Resume checks Git/package identity, source hashes, data/eval hashes, model revision, hyperparameters, seed and software versions. A completed smoke is not evidence of interrupted GPU resume; perform the explicit interruption/restart exercise and retain its logs.

Collect the archive and its hash manifest to durable storage, confirm the adapter/reload/experiment records, then **stop the rented instance via the provider**. The scripts print this reminder and never create a shutdown API key or automatically kill an unrelated machine. Artifact collection uses a strict checkpoint/adapter/metadata allowlist; existing archives are refused.

## Measurements and stop/go gates

Each actual run records GPU/driver/CUDA and package versions; sequence length, rank, precision/quantization; mask/gradient/frozen checks; input and supervised tokens; synchronized step time and tokens/s; allocated/reserved/peak VRAM; checkpoint/adapter bytes; losses and declared benchmark-pending status. `adapter_reload_validation.json` records measured logit/greedy-token agreement. CPU measurements stay labeled CPU. GPU resume success is established only by the recorded interruption/resume execution, never inferred from the tiny fixture.

Stop E1B on missing/mismatched native weights, unexpected adaptation, invalid labels, absent/nonfinite gradients, frozen gradients/weight changes, OOM or failed reload. Do not continue to E2 automatically. E2 requires rights-cleared, sufficiently rich reviewed data; current eval-family exclusions; exact-revision E0; frozen manifests; full-size modern GPU smoke/resume/reload and measured runtime; human scientific review; and explicit paid-GPU authorization with a spending limit. None is supplied by the present CPU fixture.
