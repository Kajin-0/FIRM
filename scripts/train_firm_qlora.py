#!/usr/bin/env python3
"""Train FIRM with QLoRA SFT.

Default target: compact local expert model for infrared photodetector reasoning.

Example:
    python scripts/train_firm_qlora.py \
      --model Qwen/Qwen3-4B \
      --train data/processed/firm_v2_root_expert_sft.jsonl \
      --out outputs/firm-qwen3-4b-v2

For smaller sanity runs:
    python scripts/train_firm_qlora.py \
      --model Qwen/Qwen3-0.6B \
      --train data/processed/firm_v2_root_expert_sft.jsonl \
      --out outputs/firm-qwen3-0p6b-v2 \
      --epochs 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Avoid unstable fused CCE paths on some Colab/T4/older CUDA setups.
os.environ.setdefault("UNSLOTH_USE_CCE", "0")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    data_files = {"train": str(train_path)}
    if args.valid:
        data_files["validation"] = args.valid
    ds = load_dataset("json", data_files=data_files)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

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
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
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
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved FIRM adapter to {args.out}")


if __name__ == "__main__":
    main()
