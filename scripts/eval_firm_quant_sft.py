#!/usr/bin/env python3
"""Evaluate exact-revision BF16 Qwen3.5 before/after FIRM quantitative LoRA."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from datetime import datetime,timezone
from pathlib import Path
from firm_data import sha256
from firm_model_profiles import QWEN35_REVISION,QWEN35_4B_REVISION

MODEL="Qwen/Qwen3.5-9B"

def get_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("Qwen/Qwen3.5-9B","Qwen/Qwen3.5-4B"),default=MODEL)
    p.add_argument("--data",type=Path,default=Path("data/processed/firm3_synthetic_quant_v1"))
    p.add_argument("--adapter",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-new-tokens",type=int,default=384)
    p.add_argument("--dry-run",action="store_true")
    return p.parse_args(argv)

def grade(row, raw):
    o=row["metadata"]["oracle"]
    result={"id":row["id"],"family":row["metadata"]["family"],
            "expected_value":o["value"],"expected_unit":o["unit"],
            "strict_json":False,"correct_number":False,"correct_unit":False,
            "predicted_value":None,"predicted_unit":None}
    try:
        parsed=json.loads(raw)
        if not isinstance(parsed,dict) or not isinstance(parsed.get("answer"),str) or not isinstance(parsed.get("quantities"),dict):
            return result
        result["strict_json"]=True
        quantity=parsed["quantities"].get(o["key"],{})
        if not isinstance(quantity,dict): return result
        v=quantity.get("value")
        u=quantity.get("unit")
        result["predicted_unit"]=u
        result["correct_unit"]=u==o["unit"]
        if type(v) in (int,float) and math.isfinite(v):
            result["predicted_value"]=v
            result["correct_number"]=abs(v-o["value"])<=.002*max(abs(o["value"]),1e-50)
    except (ValueError,TypeError):
        pass
    return result

def run(args):
    manifest=json.loads((args.data/"manifest.json").read_text())
    vf=args.data/"valid.jsonl"
    a=next(x for x in manifest["outputs"] if x["path"]=="valid.jsonl")
    if sha256(vf)!=a["sha256"]: raise ValueError("Validation hash changed")
    rows=[json.loads(x) for x in vf.read_text().splitlines()]
    if len(rows)!=264: raise ValueError("Unexpected validation count")
    if args.out.exists(): raise ValueError("Output exists; do not overwrite a measured comparison")
    if args.max_new_tokens not in {256,384,512}: raise ValueError("Token budget not approved")
    mode="adapter" if args.adapter else "base"
    revision=QWEN35_4B_REVISION if args.model=="Qwen/Qwen3.5-4B" else QWEN35_REVISION
    spec={"model":args.model,"revision":revision,"evaluation":"firm3_synthetic_quant_v1_valid",
          "validation_sha256":a["sha256"],"mode":mode,
          "adapter":str(args.adapter) if args.adapter else None,
          "temperature":0,"thinking":False,"max_new_tokens":args.max_new_tokens,
          "dataset_status":"same formula families as train, parameter-disjoint; not independently human reviewed"}
    if args.dry_run:
        print(json.dumps(spec,indent=2));return
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Dedicated single CUDA GPU is required")
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=revision,trust_remote_code=False)
    model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,revision=revision,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",output_loading_info=True)
    unexpected=loading.get("unexpected_keys",[])
    if any(not x.startswith("mtp.") for x in unexpected) or loading.get("missing_keys"):
        raise RuntimeError("Unexpected checkpoint loading state")
    if args.adapter:
        from peft import PeftModel
        cfg=json.loads((args.adapter/"adapter_config.json").read_text())
        if cfg.get("base_model_name_or_path")!=args.model:
            raise ValueError("Adapter base model differs")
        model=PeftModel.from_pretrained(model,str(args.adapter))
    model.eval()
    args.out.mkdir(parents=True)
    rawpath=args.out/"predictions.jsonl"
    started=time.monotonic()
    for index,row in enumerate(rows):
        prompt=tokenizer.apply_chat_template(row["messages"][:-1],tokenize=True,
                  add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                                   do_sample=False,max_new_tokens=args.max_new_tokens,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id,use_cache=True)
        answer_tokens=outputs[0,prompt.shape[1]:]
        raw=tokenizer.decode(answer_tokens,skip_special_tokens=True)
        scored=grade(row,raw)
        scored.update(raw_response=raw,output_tokens=len(answer_tokens),
                      truncated=len(answer_tokens)>=args.max_new_tokens)
        with rawpath.open("a",encoding="utf8") as f:
            f.write(json.dumps(scored,sort_keys=True,ensure_ascii=False)+"\n")
        if (index+1)%22==0:print("evaluated",index+1,"/",len(rows),flush=True)
    results=[json.loads(x) for x in rawpath.read_text().splitlines()]
    summary={**spec,"count":len(results),
             "numerical_correct":sum(x["correct_number"] for x in results),
             "unit_correct":sum(x["correct_unit"] for x in results),
             "strict_json_correct":sum(x["strict_json"] for x in results),
             "truncated":sum(x["truncated"] for x in results),
             "elapsed_seconds":time.monotonic()-started,
             "predictions_sha256":sha256(rawpath),
             "finished_utc":datetime.now(timezone.utc).isoformat()}
    summary["per_family"]={}
    for family in manifest["families"]:
        subset=[x for x in results if x["family"]==family]
        summary["per_family"][family]={
            "n":len(subset),"numerical_correct":sum(x["correct_number"] for x in subset),
            "unit_correct":sum(x["correct_unit"] for x in subset),
            "strict_json_correct":sum(x["strict_json"] for x in subset)}
    (args.out/"score.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ["mode","count","numerical_correct","unit_correct","strict_json_correct","truncated","elapsed_seconds"]},indent=2))

if __name__=="__main__":
    run(get_args())
