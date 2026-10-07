#!/usr/bin/env python3
"""FIRM quantitative LoRA training on original formula-verified synthetic examples.

A substantive adaptation experiment, distinct from the tightly gated E1B smoke.
No automatic cloud provisioning, authorization, or deployment occurs here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, subprocess
from datetime import datetime, timezone
from pathlib import Path

from firm_data import sha256, write_json, require_clean
from firm_model_profiles import (
    QWEN35_REVISION, load_model, completion_encoding, completion_collator,
    assert_adapter_boundaries, frozen_fingerprint, validate_runtime_versions
)
from firm_run_context import repository_revision

MODEL = "Qwen/Qwen3.5-9B"
EXPECTED_FAMILIES = 22

def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path,default=Path("data/processed/firm3_synthetic_quant_v1"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=660)
    p.add_argument("--max-seq-length",type=int,default=768)
    p.add_argument("--learning-rate",type=float,default=3e-5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume-from-checkpoint",type=Path)
    p.add_argument("--dry-run",action="store_true")
    return p.parse_args(argv)

def verify_data(args):
    m=json.loads((args.data/"manifest.json").read_text())
    if m.get("generator")!="build_firm_quantitative_sft.py":
        raise ValueError("Expected formula-synthetic source manifest")
    if m.get("family_count")!=EXPECTED_FAMILIES or len(m.get("families",[]))!=EXPECTED_FAMILIES:
        raise ValueError("Family mix changed")
    if not 100 <= args.steps <= 700 or args.max_seq_length not in {512,768,1024}:
        raise ValueError("Unexpected experiment budget; requires deliberate new protocol")
    if not 0 < args.learning_rate <= 1e-4:
        raise ValueError("Invalid learning rate")
    if args.resume_from_checkpoint and not args.resume_from_checkpoint.is_dir():
        raise ValueError("Explicit checkpoint directory does not exist")
    rows={}
    for split in ("train","valid"):
        path=args.data/(split+".jsonl")
        asset=next((a for a in m["outputs"] if a["path"]==path.name),None)
        if not asset or sha256(path)!=asset["sha256"]:
            raise ValueError("Corrupt or changed dataset: "+str(path))
        require_clean(path)
        rows[split]=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows[split])!=asset["rows"]:
            raise ValueError("Dataset row count changed")
        ids=set(); prompts=set(); fams=set()
        for row in rows[split]:
            meta=row["metadata"]; p=meta["provenance"]
            if meta["partition"]!=split or meta["eligibility"]!="programmatically_verified_synthetic":
                raise ValueError("Ineligible or mispartitioned synthetic row")
            if p["kind"]!="original_synthetic" or p["review_status"]!="programmatically_verified_not_human_reviewed":
                raise ValueError("Data lacks approved programmatic provenance")
            if row["id"] in ids or row["messages"][1]["content"] in prompts:
                raise ValueError("Duplicate ID/prompt in "+split)
            ids.add(row["id"]); prompts.add(row["messages"][1]["content"]); fams.add(meta["family"])
            if [x["role"] for x in row["messages"]] != ["system","user","assistant"]:
                raise ValueError("Unexpected dialogue format")
            answer=json.loads(row["messages"][2]["content"])
            key=meta["oracle"]["key"]
            if key not in answer["quantities"]:
                raise ValueError("Oracle key missing")
            q=answer["quantities"][key]
            if q["unit"]!=meta["oracle"]["unit"]:
                raise ValueError("Oracle unit changed")
            if abs(q["value"]-meta["oracle"]["value"])>6e-4*max(abs(meta["oracle"]["value"]),1e-50):
                raise ValueError("Oracle numeric mismatch")
        if fams!=set(m["families"]):
            raise ValueError("Missing training family")
    if {r["messages"][1]["content"] for r in rows["train"]} & {r["messages"][1]["content"] for r in rows["valid"]}:
        raise ValueError("Train/valid prompt collision")
    if len(rows["train"])!=2640 or len(rows["valid"])!=264:
        raise ValueError("Changed substantive experiment size")
    return m,rows

def run(args):
    manifest,rows=verify_data(args)
    run={"schema_version":"1.0","method":"bf16_lora_sft_original_formula_verified",
         "model":MODEL,"revision":QWEN35_REVISION,
         "data_manifest_sha256":sha256(args.data/"manifest.json"),
         "train_sha256":sha256(args.data/"train.jsonl"),
         "valid_sha256":sha256(args.data/"valid.jsonl"),
         "train_examples":len(rows["train"]),"validation_examples":len(rows["valid"]),
         "family_count":EXPECTED_FAMILIES,"training_steps":args.steps,
         "effective_batch_size":4,"sequence_limit":args.max_seq_length,
         "learning_rate":args.learning_rate,"rank":8,"alpha":16,"seed":args.seed,
         "source_git_revision":repository_revision(),
         "data_review":"programmatically verified, not independently human scientifically reviewed",
         "status":"PREFLIGHT"}
    if args.dry_run:
        print(json.dumps(run,indent=2)); return
    if args.out.exists() and any(args.out.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Output already exists; specify checkpoint or new run dir")
    import torch
    from firm_training_measurements import SmokeMeasurements
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig,SFTTrainer
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This substantive profile requires exactly one BF16-capable CUDA GPU")
    versions=validate_runtime_versions("qwen35_text_bf16")
    set_seed(args.seed)
    import argparse as _argparse
    loader_args=_argparse.Namespace(model=MODEL,model_revision=QWEN35_REVISION,
                                     model_profile="qwen35_text_bf16",quantization="none")
    model,tokenizer,targets=load_model(loader_args,torch.bfloat16)
    encoded={}
    for name,examples in rows.items():
        encoded[name]=Dataset.from_list([completion_encoding(tokenizer,x["messages"],args.max_seq_length)
                                         for x in examples])
    cfg=SFTConfig(output_dir=str(args.out),max_steps=args.steps,max_length=args.max_seq_length,
                  per_device_train_batch_size=1,per_device_eval_batch_size=1,
                  gradient_accumulation_steps=4,learning_rate=args.learning_rate,
                  lr_scheduler_type="cosine",warmup_steps=0.03,
                  eval_strategy="steps",eval_steps=220,save_steps=110,save_total_limit=3,
                  bf16=True,fp16=False,gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant":False},
                  completion_only_loss=True,packing=False,dataset_kwargs={"skip_prepare_dataset":True},
                  remove_unused_columns=False,optim="adamw_torch",report_to="none",
                  logging_steps=20,seed=args.seed,data_seed=args.seed)
    trainer=SFTTrainer(model=model,args=cfg,processing_class=tokenizer,
                       data_collator=completion_collator(tokenizer),
                       train_dataset=encoded["train"],eval_dataset=encoded["valid"],
                       peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,bias="none",
                                              task_type="CAUSAL_LM",target_modules=targets))
    assert_adapter_boundaries(trainer.model,True)
    start_fingerprint=frozen_fingerprint(trainer.model)
    run.update(status="TRAINING",software_versions=versions,
               gpu=torch.cuda.get_device_name(0),
               gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
               adapter_targets=targets,
               frozen_fingerprint_before=start_fingerprint,
               timestamp_utc=datetime.now(timezone.utc).isoformat())
    args.out.mkdir(parents=True,exist_ok=True)
    runfile=args.out/"firm_quant_run.json"
    if args.resume_from_checkpoint:
        prior=json.loads(runfile.read_text())
        for key in ["train_sha256","valid_sha256","model","revision","rank","alpha","training_steps","sequence_limit","learning_rate"]:
            if prior[key]!=run[key]:raise ValueError("Resume identity mismatch: "+key)
    write_json(runfile,run)
    callback=SmokeMeasurements(trainer.model,args.out,True)
    trainer.add_callback(callback)
    result=trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    callback.save()
    final_fingerprint=frozen_fingerprint(trainer.model)
    if final_fingerprint!=start_fingerprint:
        raise RuntimeError("Frozen weights changed")
    run.update(status="TRAINED",train_loss=float(result.training_loss),
               finished_step=trainer.state.global_step,
               frozen_fingerprint_after=final_fingerprint,
               finished_at_utc=datetime.now(timezone.utc).isoformat(),
               peak_vram_bytes=torch.cuda.max_memory_allocated())
    write_json(runfile,run)
    print(json.dumps({"status":run["status"],"steps":run["finished_step"],
                      "train_loss":run["train_loss"],"peak_vram_bytes":run["peak_vram_bytes"]},indent=2))

if __name__=="__main__":
    run(get_args())
