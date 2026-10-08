#!/usr/bin/env python3
"""Bounded BF16 LoRA trainer for FIRM-4B v2.3.1.

Starts from the untouched pinned Qwen3.5-4B base. v2.3.1 rebalances the v2.3 curriculum toward natural-prompt complete answers,
multi-turn expansion, and short-thinking/long-final calibration while preserving
broad specialist and quantitative coverage. It never imports private benchmark prompts/gold.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from firm_model_profiles import QWEN35_4B_REVISION, validate_runtime_versions, load_model
from train_firm4b_v2_1 import (
    MODEL, frozen_fingerprint, completion_encoding, completion_collator,
    assert_adapter_boundaries, sha256, repository_revision, write_json
)

DATA_DEFAULT=ROOT/"data/processed/firm4b_v2_3_1_expanded_v1"

def get_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=DATA_DEFAULT)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1373)
    p.add_argument("--max-seq-length",type=int,default=1024)
    p.add_argument("--learning-rate",type=float,default=1.5e-5)
    p.add_argument("--seed",type=int,default=20261010)
    p.add_argument("--resume-from-checkpoint",type=Path)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--allow-provisional-data",action="store_true")
    return p.parse_args(argv)

def _conversation_key(messages):
    reasoning=messages[-1].get("reasoning_content") if messages else None
    thinking=isinstance(reasoning,str) and bool(reasoning.strip())
    return json.dumps({"thinking":thinking,"prefix":messages[:-1]},ensure_ascii=False,sort_keys=True)

def _valid_roles(messages):
    roles=[m.get("role") for m in messages]
    if len(roles)<3 or roles[0]!="system" or roles[-1]!="assistant":
        return False
    expected="user"
    for role in roles[1:]:
        if role!=expected:
            return False
        expected="assistant" if expected=="user" else "user"
    return True

def verify_data(args):
    m=json.loads((args.data/"manifest.json").read_text())
    if m.get("schema_version")!="2.3.1": raise ValueError("Unexpected v2.3.1 manifest")
    if m.get("private_benchmark_imported") is not False or m.get("exact_user_stress_prompts_imported") is not False:
        raise ValueError("Benchmark/stress contamination flag")
    if "PROVISIONAL" in m.get("status","") and not args.allow_provisional_data:
        raise PermissionError("v2.3.1 data is research-provisional; use --allow-provisional-data explicitly")
    if not (200<=args.steps<=1500): raise ValueError("v2.3.1 bounded steps must be 200..1500")
    if args.max_seq_length not in {768,1024,1536}: raise ValueError("unsupported sequence length")
    if not (0 < args.learning_rate <= 3e-5): raise ValueError("v2.3.1 LR must be <=3e-5")
    rows={}
    for split in ("train","valid"):
        p=args.data/(split+".jsonl")
        if sha256(p)!=m[split+"_sha256"]: raise ValueError(split+" hash mismatch")
        rows[split]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len(rows[split])!=m["rows"][split]: raise ValueError(split+" count mismatch")
        keys=set()
        for row in rows[split]:
            msgs=row.get("messages",[])
            if not _valid_roles(msgs): raise ValueError("Expected alternating system/user/assistant conversation")
            key=_conversation_key(msgs)
            if key in keys: raise ValueError("duplicate "+split+" conversation")
            keys.add(key)
            if not msgs[-1].get("content","").strip(): raise ValueError("empty final target")
    train_keys={_conversation_key(x["messages"]) for x in rows["train"]}
    valid_keys={_conversation_key(x["messages"]) for x in rows["valid"]}
    if train_keys & valid_keys: raise ValueError("train/valid conversation overlap")
    all_user_turns={m["content"] for split in rows.values() for r in split for m in r["messages"] if m.get("role")=="user"}
    for ep in [ROOT/"evals/firm_specialist_scope_v1.jsonl",ROOT/"evals/firm_stress_v1.jsonl",
               ROOT/"evals/firm_v22_gauntlet_v1.jsonl",ROOT/"evals/firm_v23_depth_gauntlet_v1.jsonl",
               ROOT/"evals/firm_v231_shadow_gauntlet_v1.jsonl"]:
        if not ep.exists(): continue
        held={json.loads(x).get("prompt") for x in ep.read_text().splitlines() if x.strip()}
        held.discard(None)
        if all_user_turns & held: raise ValueError("public evaluation prompt leakage")
    return m,rows

def completion_encoding_v231(tokenizer,messages,max_length):
    """Mask the complete conversation prefix and supervise only the final assistant turn."""
    if not _valid_roles(messages):
        raise ValueError("Invalid alternating conversation roles")
    if any(any(t in m.get("content","") for t in ("<|image_pad|>","<|video_pad|>","<|vision_start|>")) for m in messages):
        raise ValueError("Text-only profile forbids visual placeholders")
    assistant=messages[-1]
    reasoning=assistant.get("reasoning_content")
    thinking=isinstance(reasoning,str) and bool(reasoning.strip())
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,
        add_generation_prompt=True,enable_thinking=thinking)
    full=tokenizer.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=False,enable_thinking=thinking)
    if not full.startswith(prefix):
        raise ValueError("Chat-template completion boundary changed")
    ids=tokenizer.encode(full,add_special_tokens=False)
    prompt_ids=tokenizer.encode(prefix,add_special_tokens=False)
    if ids[:len(prompt_ids)]!=prompt_ids:
        raise ValueError("Tokenizer merges across final completion boundary")
    if len(ids)>max_length or len(prompt_ids)>=len(ids):
        raise ValueError("Overlength or empty final target")
    return {"input_ids":ids,"attention_mask":[1]*len(ids),
            "labels":[-100]*len(prompt_ids)+ids[len(prompt_ids):]}

def run(args):
    manifest,rows=verify_data(args)
    runmeta={"schema_version":"2.3.1","method":"bf16_lora_firm4b_v2_3_1",
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
        raise RuntimeError("v2.3.1 requires exactly one BF16 CUDA GPU")
    versions=validate_runtime_versions("qwen35_4b_text_bf16")
    set_seed(args.seed)
    import argparse as _ap
    loader=_ap.Namespace(model=MODEL,model_revision=QWEN35_4B_REVISION,
                         model_profile="qwen35_4b_text_bf16",quantization="none")
    model,tokenizer,targets=load_model(loader,torch.bfloat16)
    encoded={k:Dataset.from_list([completion_encoding_v231(tokenizer,x["messages"],args.max_seq_length) for x in v])
             for k,v in rows.items()}
    cfg=SFTConfig(output_dir=str(args.out),max_steps=args.steps,max_length=args.max_seq_length,
      per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,
      learning_rate=args.learning_rate,lr_scheduler_type="cosine",warmup_steps=0.03,
      eval_strategy="steps",eval_steps=275,save_steps=125,save_total_limit=4,
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
    runfile=args.out/"firm_v231_run.json"
    write_json(runfile,runmeta)
    cb=SmokeMeasurements(trainer.model,args.out,True);trainer.add_callback(cb)
    result=trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(args.out);tokenizer.save_pretrained(args.out)
    # Preserve the exact pinned base's reload-compatible tokenizer declaration.
    # Transformers may otherwise serialize the runtime alias Qwen3_5Tokenizer,
    # which older/local conversion environments cannot import by class name.
    tcfg_path=args.out/"tokenizer_config.json"
    tcfg=json.loads(tcfg_path.read_text())
    tcfg["tokenizer_class"]="Qwen2Tokenizer"
    if getattr(tokenizer,"chat_template",None):
        tcfg["chat_template"]=tokenizer.chat_template
        (args.out/"chat_template.jinja").write_text(tokenizer.chat_template)
    tcfg_path.write_text(json.dumps(tcfg,indent=2,ensure_ascii=False)+"\n")
    cb.save()
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
