#!/usr/bin/env python3
"""Score private FIRM quantitative challenge without disclosing held-out gold answers.

Input JSONL records: {"id":"...","response":"<model answer>","finish_reason":"stop"}.
No model is invoked here. Supports paired model-vs-model comparisons by preserving
the same locked question IDs and manifests. Do not train on benchmark outputs.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

PRIVATE_ROOT=Path.home()/".local/share/firm-private-bench/large_quant_v1"

def parse_answer(response):
    if not isinstance(response,str):return None
    try: answer=json.loads(response)
    except (json.JSONDecodeError,ValueError):return None
    if not isinstance(answer,dict):return None
    if not isinstance(answer.get("quantities"),dict):return None
    return answer["quantities"]

def grade(gold,pred):
    result={"response_present":False,"json_valid":False,"number_correct":False,
            "unit_correct":False,"joint_correct":False,"truncated":False,
            "refusal":False}
    if pred is None:return result
    response=pred.get("response",pred.get("content",""))
    if not isinstance(response,str):return result
    result["response_present"]=bool(response.strip())
    result["truncated"]=pred.get("finish_reason")=="length"
    result["refusal"]=bool(pred.get("refusal",False))
    q=parse_answer(response)
    if q is None:return result
    result["json_valid"]=True
    specs=gold["grading"]["quantities"]
    numbers=[]; units=[]
    for key,expected in specs.items():
        answer=q.get(key)
        if not isinstance(answer,dict):
            numbers.append(False);units.append(False);continue
        value=answer.get("value")
        good_numeric=type(value) in (int,float) and math.isfinite(value)
        if good_numeric:
            good_numeric=math.isclose(value,expected["value"],
                                      rel_tol=expected["rtol"],abs_tol=expected["atol"])
        numbers.append(good_numeric)
        unit=answer.get("unit")
        units.append(isinstance(unit,str) and unit.strip().replace("µ","u").replace("μ","u") == expected["unit"].replace("µ","u").replace("μ","u"))
    result["number_correct"]=bool(numbers) and all(numbers)
    result["unit_correct"]=bool(units) and all(units)
    result["joint_correct"]=result["number_correct"] and result["unit_correct"]
    return result

def wilson(successes,total,z=1.959963984540054):
    if total==0:return [0.,1.]
    p=successes/total
    den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [0. if successes==0 else max(0.,center-radius),
            1. if successes==total else min(1.,center+radius)]

def score(bank, predfile, tier,allow_partial=False):
    mf=json.loads((bank/"manifest.json").read_text())
    asset=next(a for a in mf["tiers"] if a["name"]==tier)
    goldfile=bank/asset["path"]
    if hashlib.sha256(goldfile.read_bytes()).hexdigest()!=asset["sha256"]:
        raise ValueError("Private benchmark gold hash mismatch")
    golds={};families=collections.defaultdict(list)
    for raw in goldfile.read_text().splitlines():
        record=json.loads(raw)
        if record["id"] in golds: raise ValueError("Duplicate gold id")
        golds[record["id"]]=record
    predictions={}
    for raw in predfile.read_text().splitlines():
        if not raw.strip():continue
        record=json.loads(raw)
        id_=record.get("id")
        if id_ in predictions:raise ValueError("Duplicate prediction id: "+str(id_))
        if id_ not in golds:raise ValueError("Unknown question id: "+str(id_))
        predictions[id_]=record
    if not allow_partial and set(predictions)!=set(golds):
        raise ValueError(f"Incomplete evaluation {len(predictions)}/{len(golds)}; --allow-partial for diagnostics only")
    results={}
    for id_,g in golds.items():
        if id_ not in predictions:continue
        v=grade(g,predictions[id_])
        results[id_]=v
        families[g["family_id"]].append(v)
    keys=("response_present","json_valid","number_correct","unit_correct","joint_correct","truncated","refusal")
    def summarize(recs):
        d={"count":len(recs)}
        for key in keys:
            n=sum(bool(r[key]) for r in recs)
            d[key]={"n":n,"rate":n/len(recs) if recs else 0.,
                    "wilson_95":wilson(n,len(recs))}
        return d
    return {"schema_version":"1.0","bank_commitment":asset["sha256"],"tier":tier,
            "predictions_sha256":hashlib.sha256(predfile.read_bytes()).hexdigest(),
            "completed":len(predictions)==len(golds),
            "coverage":len(predictions)/len(golds),
            "overall":summarize(list(results.values())),
            "per_family":{k:summarize(v) for k,v in sorted(families.items())},
            "caution":"Parameter holdout from known families does not establish reasoning transfer; unseen-family tier is separate. Confidence intervals are item-level and can understate dependence from shared formula templates."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",type=Path,default=PRIVATE_ROOT)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--tier",choices=("known_family_parameter_holdout","unseen_family_transfer"),required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--allow-partial",action="store_true")
    args=p.parse_args()
    result=score(args.bank,args.predictions,args.tier,args.allow_partial)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise FileExistsError("Do not overwrite a benchmark result")
        args.output.write_text(text)
    print(text)

if __name__=="__main__":
    main()
