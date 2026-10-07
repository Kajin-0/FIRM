#!/usr/bin/env python3
"""FIRM-4B v2.1 expanded specialist SFT from untouched pinned Qwen3.5-4B base.

v2 changes vs v1:
- rebalanced original curriculum: canonical IR facts + natural quantitative reasoning
- much lower refusal density
- typo/paraphrase positive in-scope examples
- lower LR to reduce destructive domain regressions
- rank-16 MLP LoRA for additional specialist capacity
No cloud provisioning occurs here.
"""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from firm_data import sha256,write_json
from firm_model_profiles import (QWEN35_4B_REVISION,load_model,completion_encoding,
    completion_collator,assert_adapter_boundaries,frozen_fingerprint,validate_runtime_versions)
from firm_run_context import repository_revision
MODEL="Qwen/Qwen3.5-4B"
DATA_DEFAULT=Path("data/processed/firm4b_v2_foundation_v2")

def get_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=DATA_DEFAULT)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1250)
    p.add_argument("--max-seq-length",type=int,default=1024)
    p.add_argument("--learning-rate",type=float,default=1.5e-5)
    p.add_argument("--seed",type=int,default=20261007)
    p.add_argument("--resume-from-checkpoint",type=Path)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--allow-provisional-data",action="store_true")
    return p.parse_args(argv)

def verify_data(args):
    m=json.loads((args.data/"manifest.json").read_text())
    if m.get("schema_version")!="2.1": raise ValueError("Unexpected v2 manifest")
    if m.get("private_benchmark_imported") is not False or m.get("exact_user_stress_prompts_imported") is not False:
        raise ValueError("Benchmark/stress contamination flag")
    if "PROVISIONAL" in m.get("status","") and not args.allow_provisional_data:
        raise PermissionError("v2 data is research-provisional; use --allow-provisional-data explicitly")
    if not (200<=args.steps<=1400): raise ValueError("v2 bounded steps must be 200..1400")
    if args.max_seq_length not in {768,1024,1536}: raise ValueError("unsupported sequence length")
    if not (0 < args.learning_rate <= 3e-5): raise ValueError("v2 LR must be <=3e-5")
    rows={}
    for split in ("train","valid"):
        p=args.data/(split+".jsonl")
        expected=m[split+"_sha256"]
        if sha256(p)!=expected: raise ValueError(split+" hash mismatch")
        rows[split]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len(rows[split])!=m["rows"][split]: raise ValueError(split+" count mismatch")
        prompts=set()
        for row in rows[split]:
            msgs=row.get("messages",[])
            if [x.get("role") for x in msgs] != ["system","user","assistant"]:
                raise ValueError("Expected system/user/assistant messages")
            prompt=msgs[1]["content"]
            if prompt in prompts: raise ValueError("duplicate "+split+" prompt")
            prompts.add(prompt)
            if not msgs[2]["content"].strip(): raise ValueError("empty target")
    train_prompts={x["messages"][1]["content"] for x in rows["train"]}
    valid_prompts={x["messages"][1]["content"] for x in rows["valid"]}
    if train_prompts & valid_prompts: raise ValueError("train/valid prompt overlap")
    # Public evaluation prompts are excluded exactly; private benchmark is never read here.
    for ep in [Path("evals/firm_specialist_scope_v1.jsonl"),Path("evals/firm_stress_v1.jsonl")]:
        if not ep.exists(): continue
        held={json.loads(x).get("prompt") for x in ep.read_text().splitlines() if x.strip()}
        held.discard(None)
        if (train_prompts|valid_prompts)&held: raise ValueError("public evaluation prompt leakage")
    return m,rows

def run(args):
    manifest,rows=verify_data(args)
    runmeta={"schema_version":"2.0","method":"bf16_lora_firm4b_v2_1_expanded",
      "model":MODEL,"revision":QWEN35_4B_REVISION,
      "data_manifest_sha256":sha256(args.data/"manifest.json"),
      "train_sha256":sha256(args.data/"train.jsonl"),"valid_sha256":sha256(args.data/"valid.jsonl"),
      "train_examples":len(rows["train"]),"validation_examples":len(rows["valid"]),
      "training_steps":args.steps,"effective_batch_size":4,"sequence_limit":args.max_seq_length,
      "learning_rate":args.learning_rate,"rank":16,"alpha":32,"seed":args.seed,
      "source_git_revision":repository_revision(),"data_review":manifest["status"],"status":"PREFLIGHT"}
    if args.dry_run:
        print(json.dumps(runmeta,indent=2));return
    if args.out.exists() and any(args.out.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Output exists; use new directory or resume")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import SFTConfig,SFTTrainer
    from firm_training_measurements import SmokeMeasurements
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("v2 requires exactly one BF16 CUDA GPU")
    versions=validate_runtime_versions("qwen35_4b_text_bf16")
    set_seed(args.seed)
    import argparse as _ap
    loader=_ap.Namespace(model=MODEL,model_revision=QWEN35_4B_REVISION,
                         model_profile="qwen35_4b_text_bf16",quantization="none")
    model,tokenizer,targets=load_model(loader,torch.bfloat16)
    encoded={k:Dataset.from_list([completion_encoding(tokenizer,x["messages"],args.max_seq_length) for x in v])
             for k,v in rows.items()}
    cfg=SFTConfig(output_dir=str(args.out),max_steps=args.steps,max_length=args.max_seq_length,
      per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,
      learning_rate=args.learning_rate,lr_scheduler_type="cosine",warmup_steps=0.03,
      eval_strategy="steps",eval_steps=250,save_steps=125,save_total_limit=4,
      bf16=True,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
      completion_only_loss=True,packing=False,dataset_kwargs={"skip_prepare_dataset":True},
      remove_unused_columns=False,optim="adamw_torch",report_to="none",logging_steps=20,
      seed=args.seed,data_seed=args.seed)
    trainer=SFTTrainer(model=model,args=cfg,processing_class=tokenizer,
      data_collator=completion_collator(tokenizer),train_dataset=encoded["train"],
      eval_dataset=encoded["valid"],
      peft_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,bias="none",
                             task_type="CAUSAL_LM",target_modules=targets))
    assert_adapter_boundaries(trainer.model,True)
    before=frozen_fingerprint(trainer.model)
    runmeta.update(status="TRAINING",software_versions=versions,gpu=torch.cuda.get_device_name(0),
      gpu_bytes=torch.cuda.get_device_properties(0).total_memory,adapter_targets=targets,
      frozen_fingerprint_before=before,timestamp_utc=datetime.now(timezone.utc).isoformat())
    args.out.mkdir(parents=True,exist_ok=True)
    runfile=args.out/"firm_v2_run.json"
    if args.resume_from_checkpoint:
        prior=json.loads(runfile.read_text())
        for key in ["train_sha256","valid_sha256","model","revision","rank","alpha",
                    "training_steps","sequence_limit","learning_rate"]:
            if prior[key]!=runmeta[key]: raise ValueError("Resume identity mismatch: "+key)
    write_json(runfile,runmeta)
    cb=SmokeMeasurements(trainer.model,args.out,True);trainer.add_callback(cb)
    result=trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(args.out);tokenizer.save_pretrained(args.out);cb.save()
    after=frozen_fingerprint(trainer.model)
    if after!=before: raise RuntimeError("Frozen weights changed")
    runmeta.update(status="TRAINED",train_loss=float(result.training_loss),
      finished_step=trainer.state.global_step,frozen_fingerprint_after=after,
      finished_at_utc=datetime.now(timezone.utc).isoformat(),
      peak_vram_bytes=torch.cuda.max_memory_allocated())
    write_json(runfile,runmeta)
    print(json.dumps({"status":"TRAINED","steps":trainer.state.global_step,
                      "train_loss":float(result.training_loss),
                      "peak_vram_bytes":runmeta["peak_vram_bytes"]},indent=2))
if __name__=="__main__": run(get_args())
