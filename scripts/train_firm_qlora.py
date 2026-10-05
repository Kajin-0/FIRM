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
import signal
from datetime import datetime, timezone
from pathlib import Path

from firm_data import require_clean, sha256, write_json
from firm_model_profiles import validate_profile, validate_runtime_versions, load_model, completion_encoding, completion_collator, assert_adapter_boundaries, frozen_fingerprint
from firm_run_context import repository_revision


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
    ap.add_argument('--model-profile',choices=['legacy_causal','qwen35_text_bf16'],default='legacy_causal')
    ap.add_argument('--quantization',choices=['nf4','none'],default='nf4')
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
    validate_profile(args)
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
            if args.model_profile=='qwen35_text_bf16' and row['metadata']['provenance']['review_status'] not in {'AI_analytically_reviewed','programmatically_corrected'}:
                raise ValueError('Modern smoke requires explicitly reviewed examples')
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
            "git_sha": repository_revision(),
            "base_model": args.model, "base_model_revision": args.model_revision, "python_version": platform.python_version(),
            "config": {"path": str(args.config), "sha256": sha256(args.config)} if args.config else None,
            "dataset_manifest": {"path": str(args.dataset_manifest), "sha256": sha256(args.dataset_manifest)},
            "eval_manifests": evals, "training_method": "qlora_sft_completion_only" if args.quantization=='nf4' else 'bf16_lora_sft_completion_only',
            'model_profile':args.model_profile,'quantization':args.quantization,
            'tool_sha256':{p:sha256(Path(__file__).with_name(p)) for p in ['train_firm_qlora.py','firm_model_profiles.py','firm_training_measurements.py','firm_data.py','firm_run_context.py']},
            "hyperparameters": {k: getattr(args, k) for k in ("max_seq_length", "batch_size", "grad_accum", "lr",
                "epochs", "lora_r", "lora_alpha", "lora_dropout", "eval_steps", "save_steps", "max_steps", "max_train_examples")},
            "seed": args.seed, "checkpoint_path": str(out), "adapter_path": str(out),
            "training_loss": None, "validation_loss": None, "benchmark_results": None,
            "notes": "Pipeline smoke only. Scientific review status is explicit in dataset lineage; no detector-quality claim."}


def require_reviewed_inputs(args):
    for path,cap in [(Path(args.train),args.max_train_examples),(Path(args.valid) if args.valid else None,None)]:
        if path is None:continue
        rows=require_clean(path)
        if cap:rows=rows[:cap]
        for row in rows:
            if row['metadata'].get('provenance',{}).get('review_status') not in {'AI_analytically_reviewed','programmatically_corrected'}:
                raise ValueError('Actual training refuses scientifically unreviewed data: '+str(path))
            if row['metadata'].get('eligibility')!='reviewed_science_candidate':
                raise ValueError('Actual training refuses quarantined or excluded rows')


def main() -> None:
    args = parse_args()
    run = preflight(args)
    if args.dry_run:
        print(json.dumps({**run, "status": "dry_run_no_training"}, indent=2))
        return
    require_reviewed_inputs(args)
    validate_runtime_versions(args.model_profile)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
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

    model,tokenizer,targets = load_model(args,compute_dtype)
    modern=args.model_profile=='qwen35_text_bf16'
    if modern:
        ds={'train':Dataset.from_list([completion_encoding(tokenizer,r['messages'],args.max_seq_length)
                                      for r in require_clean(train_path)[:args.max_train_examples]])}
        if args.valid:
            ds['validation']=Dataset.from_list([completion_encoding(tokenizer,r['messages'],args.max_seq_length)
                                               for r in require_clean(args.valid)])
    for partition in ds:
        for row in ds[partition]:
            length = len(row['input_ids']) if modern else len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True))
            if length > args.max_seq_length:
                raise ValueError(f"{partition} example has {length} tokens; increase sequence length or curate explicitly instead of truncating")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    sft_args = SFTConfig(
        output_dir=args.out,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        packing=False,
        dataset_kwargs={'skip_prepare_dataset':True} if modern else None,
        remove_unused_columns=False if modern else True,
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
        gradient_checkpointing_kwargs={'use_reentrant':False} if modern else None,
        optim="adamw_torch" if modern else "paged_adamw_8bit",
        bf16=bf16,
        fp16=fp16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=completion_collator(tokenizer) if modern else None,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_config,
    )

    trainable=assert_adapter_boundaries(trainer.model,modern)
    before_frozen=frozen_fingerprint(trainer.model)
    probe_batch=next(iter(trainer.get_train_dataloader()))
    if not (probe_batch['labels']==-100).any() or not (probe_batch['labels']!=-100).any():
        raise ValueError('Completion mask must include both masked prompt and supervised answer tokens')
    run['completion_mask_check']={'masked_tokens':int((probe_batch['labels']==-100).sum()),
                                  'supervised_tokens':int((probe_batch['labels']!=-100).sum())}
    run['adapter_targets']=targets
    run['checkpoint_loading_info']=getattr(model,'firm_loading_info',None)
    run['trainable_parameter_count']=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    from firm_training_measurements import SmokeMeasurements
    measurements=SmokeMeasurements(trainer.model,Path(args.out),modern)
    trainer.add_callback(measurements)
    signal.signal(signal.SIGTERM,lambda signum,frame:setattr(measurements,'interruption_requested',True))
    run["timestamp"] = datetime.now(timezone.utc).isoformat()
    run["hardware"] = {"gpu": torch.cuda.get_device_name(0), "gpu_count": 1, "cuda": torch.version.cuda,
                       "precision": "bf16" if bf16 else "fp16", "quantization": args.quantization,
                       'total_vram_bytes':torch.cuda.get_device_properties(0).total_memory,
                       'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    run["software_versions"] = {p: importlib.metadata.version(p) for p in
        ("torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes")}
    manifest_path = Path(args.out) / "experiment_manifest.json"
    if args.resume_from_checkpoint:
        previous = json.loads(manifest_path.read_text())
        for key in ("run_id", "git_sha", "base_model", "base_model_revision", "dataset_manifest", "eval_manifests", "hyperparameters", "seed", 'model_profile','quantization','tool_sha256','software_versions'):
            if previous[key] != run[key]:
                raise ValueError("Resume changes recorded experiment variable: " + key)
        if not args.resume_from_checkpoint.is_dir():
            raise ValueError("Resume checkpoint directory missing")
    run["status"] = "training"
    write_json(manifest_path, run)
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    if measurements.interruption_requested and trainer.state.global_step < args.max_steps:
        run.update(status='interrupted_checkpoint_saved',global_step=trainer.state.global_step)
        write_json(manifest_path,run)
        print('SIGTERM: checkpoint saved at optimizer boundary; resume explicitly')
        return
    if frozen_fingerprint(trainer.model)!=before_frozen:
        raise ValueError('Sampled frozen parameters changed; do not treat smoke as passed')
    run['frozen_parameter_check']='all frozen requires_grad=False; no frozen gradients; every frozen tensor sampled unchanged'
    trainer.model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=compute_dtype):
        probe_input=probe_batch['input_ids'][:1]
        probe_mask=probe_batch['attention_mask'][:1]
        logits=trainer.model(input_ids=probe_input,attention_mask=probe_mask,use_cache=False).logits[0,-1].float().cpu()
    indices=list(range(0,len(logits),max(1,len(logits)//64)))
    write_json(Path(args.out)/'reload_reference.json',{'input_ids':probe_input.cpu().tolist(),
        'attention_mask':probe_mask.cpu().tolist(),'logit_indices':indices,
        'last_logits':logits[indices].tolist(),'greedy_next_token':int(logits.argmax()),'rtol':.005,'atol':.005,
        'autocast_dtype':'bfloat16' if bf16 else 'float16'})
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    measurements.save()
    run.update({"status": "training_complete_benchmark_pending", "training_loss": result.metrics.get("train_loss"),
                "validation_loss": trainer.evaluate().get("eval_loss") if "validation" in ds else None,
                "trainer_metrics": result.metrics})
    write_json(manifest_path, run)
    print(f"Saved FIRM adapter to {args.out}")


if __name__ == "__main__":
    main()
