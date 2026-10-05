#!/usr/bin/env python3
"""Pinned, single-GPU causal QLoRA smoke/control path for FIRM.

Validate the prepared candidate artifacts without GPU imports or downloads:
    python scripts/train_firm_qlora.py --config configs/firm3_e1_smoke.json --dry-run

An actual run requires a reviewed smoke subset and a compatible CUDA environment;
see docs/FIRM3_TRAINING_PLAN.md before removing --dry-run. Modern vision/hybrid
candidates require a separately validated training profile.
"""

from __future__ import annotations

import argparse
import json
import importlib.metadata
import re
import subprocess
import platform
from datetime import datetime, timezone
from pathlib import Path

from firm_data import require_clean, sha256, write_json


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, help="JSON profile; command-line flags override defaults")
    config_path = pre.parse_known_args()[0].config
    config = json.loads(config_path.read_text()) if config_path else {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--train", default="data/processed/firm_v2_root_expert_sft.jsonl")
    ap.add_argument("--valid", default=None)
    ap.add_argument("--out", default="outputs/firm-qwen3-4b-v2")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--model-revision", required="model_revision" not in config, help="Immutable 40-character model commit SHA")
    ap.add_argument("--dataset-manifest", required="dataset_manifest" not in config, type=Path)
    ap.add_argument("--eval-manifest", required="eval_manifest" not in config, action="append", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-train-examples", type=int)
    ap.add_argument("--resume-from-checkpoint", type=Path)
    ap.add_argument("--run-id", required="run_id" not in config)
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and manifest without GPU imports or downloads")
    unknown = set(config) - {action.dest for action in ap._actions}
    if unknown:
        raise ValueError("Unknown config keys: " + str(sorted(unknown)))
    ap.set_defaults(**config)
    args = ap.parse_args()
    args.dataset_manifest = Path(args.dataset_manifest)
    args.eval_manifest = [Path(p) for p in args.eval_manifest]
    args.resume_from_checkpoint = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    return args


def preflight(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("Pin --model-revision to an immutable model commit SHA")
    if args.model.startswith(("Qwen/Qwen3.5", "Qwen/Qwen3.6", "Qwen/Qwen3.8", "google/gemma-4", "mistralai/Ministral-3")):
        raise ValueError("This pinned smoke trainer is for the legacy causal stack; VLM/hybrid training needs a separately validated profile")
    if args.max_seq_length < 32 or args.batch_size < 1 or args.grad_accum < 1 or args.lora_r < 1 or args.lr <= 0:
        raise ValueError("Invalid sequence length, batch size, accumulation, rank or learning rate")
    if args.max_steps == 0 or args.max_steps < -1 or args.save_steps < 1 or args.eval_steps < 1:
        raise ValueError("Invalid step settings")
    if args.max_train_examples is not None and args.max_train_examples < 1:
        raise ValueError("Invalid example cap")
    manifest = json.loads(args.dataset_manifest.read_text())
    for asset in manifest["outputs"]:
        if sha256(args.dataset_manifest.parent / asset["path"]) != asset["sha256"]:
            raise ValueError("Dataset hash mismatch: " + asset["path"])
    train, valid = Path(args.train), Path(args.valid) if args.valid else None
    for path, partition in [(train, "train"), (valid, "valid")]:
        if path is None:
            continue
        if path.resolve() != (args.dataset_manifest.parent / (partition + ".jsonl")).resolve():
            raise ValueError("Use the declared train/valid partition from the dataset manifest")
        for row in require_clean(path):
            if row["metadata"].get("partition") != partition:
                raise ValueError("Wrong partition in " + str(path))
            if sum(m["role"] == "assistant" for m in row.get("messages", [])) != 1:
                raise ValueError("Smoke trainer requires a single assistant completion")
            if not row["messages"] or row["messages"][-1]["role"] != "assistant":
                raise ValueError("Smoke trainer requires a final assistant completion")
    evals = []
    for path in args.eval_manifest:
        obj = json.loads(path.read_text())
        for asset in obj["assets"]:
            if sha256(asset["path"]) != asset["sha256"]:
                raise ValueError("Eval hash mismatch: " + asset["path"])
        evals.append({"path": str(path), "sha256": sha256(path)})
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Output contains work; use a new run path or explicit resume checkpoint")
    return {"schema_version": "1.0", "run_id": args.run_id,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "base_model": args.model, "base_model_revision": args.model_revision, "python_version": platform.python_version(),
            "config": {"path": str(args.config), "sha256": sha256(args.config)} if args.config else None,
            "dataset_manifest": {"path": str(args.dataset_manifest), "sha256": sha256(args.dataset_manifest)},
            "eval_manifests": evals, "training_method": "qlora_sft_completion_only",
            "hyperparameters": {k: getattr(args, k) for k in ("max_seq_length", "batch_size", "grad_accum", "lr",
                "epochs", "lora_r", "lora_alpha", "lora_dropout", "eval_steps", "save_steps", "max_steps", "max_train_examples")},
            "seed": args.seed, "checkpoint_path": str(out), "adapter_path": str(out),
            "training_loss": None, "validation_loss": None, "benchmark_results": None,
            "notes": "Candidate data remains unreviewed; this profile is a pipeline smoke/controlled-ablation path."}


def main() -> None:
    args = parse_args()
    run = preflight(args)
    if args.dry_run:
        print(json.dumps({**run, "status": "dry_run_no_training"}, indent=2))
        return
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA profile needs an available CUDA GPU; use --dry-run on the VPS")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This validated-design profile uses one visible GPU; multi-GPU training is planned separately")
    set_seed(args.seed)
    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    # Conversational prompt-completion gives completion-only loss without requiring
    # template-specific assistant masks. Provenance stays in the hashed source.
    def dataset(path, cap=None):
        rows = require_clean(path)
        if cap:
            rows = rows[:cap]
        return Dataset.from_list([{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]}
                                  for r in rows])
    ds = {"train": dataset(train_path, args.max_train_examples)}
    if args.valid:
        ds["validation"] = dataset(Path(args.valid))

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=False,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    for partition in ds:
        for row in ds[partition]:
            length = len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True))
            if length > args.max_seq_length:
                raise ValueError(f"{partition} example has {length} tokens; increase sequence length or curate explicitly instead of truncating")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_args = SFTConfig(
        output_dir=args.out,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps" if "validation" in ds else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=fp16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_config,
    )

    run["timestamp"] = datetime.now(timezone.utc).isoformat()
    run["hardware"] = {"gpu": torch.cuda.get_device_name(0), "gpu_count": 1, "cuda": torch.version.cuda,
                       "precision": "bf16" if bf16 else "fp16", "quantization": "NF4 double quantization"}
    run["software_versions"] = {p: importlib.metadata.version(p) for p in
        ("torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes")}
    manifest_path = Path(args.out) / "experiment_manifest.json"
    if args.resume_from_checkpoint:
        previous = json.loads(manifest_path.read_text())
        for key in ("run_id", "git_sha", "base_model", "base_model_revision", "dataset_manifest", "eval_manifests", "hyperparameters", "seed"):
            if previous[key] != run[key]:
                raise ValueError("Resume changes recorded experiment variable: " + key)
        if not args.resume_from_checkpoint.is_dir():
            raise ValueError("Resume checkpoint directory missing")
    run["status"] = "training"
    write_json(manifest_path, run)
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    run.update({"status": "training_complete_benchmark_pending", "training_loss": result.metrics.get("train_loss"),
                "validation_loss": trainer.evaluate().get("eval_loss") if "validation" in ds else None,
                "trainer_metrics": result.metrics})
    write_json(manifest_path, run)
    print(f"Saved FIRM adapter to {args.out}")


if __name__ == "__main__":
    main()
