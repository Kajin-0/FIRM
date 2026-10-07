#!/usr/bin/env python3
"""Assemble a scientifically review-gated FIRM 4B SFT dataset.

Mix: 2640 independently formula-checked numeric rows; each approved conceptual
training example duplicated 3x; each approved out-of-scope example 2x.
No SFT may be emitted until ALL editorial cards are independently audited.
"""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
from review_firm_specialist_cards import inspect

ROOT=Path(__file__).resolve().parents[1]
QUANT=ROOT/"data/processed/firm3_synthetic_quant_v1"
CONCEPT=ROOT/"data/processed/firm3_specialist_concepts_v1"
REVIEW=ROOT/"data/reviews/firm3_specialist_concepts_v1_review.csv"
EVAL=ROOT/"evals/firm_specialist_scope_v1.jsonl"

def h(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_checked(path,name,split):
    manifest=json.loads((path/"manifest.json").read_text())
    expected=manifest["outputs"][split]["sha256"] if name=="concept" else next(
        x["sha256"] for x in manifest["outputs"] if x["path"]==split+".jsonl")
    file=path/(split+".jsonl")
    if h(file)!=expected:raise ValueError("Content hash mismatch: "+str(file))
    return [json.loads(s) for s in file.read_text().splitlines() if s.strip()]

def summary():
    audit=inspect(REVIEW)
    return {"model":"Qwen/Qwen3.5-4B","revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "review":{k:audit[k] for k in ["total","approved","blocked"]},
            "training_permission":audit["blocked"]==0,
            "mix_if_approved":{"train":{"numeric":2640,"concept":1071,"scope":600,"total":4311},
                               "valid":{"numeric":264,"concept":51,"scope":50,"total":365}},
            "note":"Generating training data blocked until all 101 independent card approvals complete"}

def build(out,provisional=False):
    a=summary()
    if not a["training_permission"] and not provisional:
        raise PermissionError("101-card scientific/scope review gate not passed: "
                              +str(a["review"]["blocked"])+" blocked; exploratory training requires --provisional-research")
    if out.exists():raise FileExistsError("Output already exists: "+str(out))
    heldout={json.loads(s)["prompt"] for s in EVAL.read_text().splitlines()}
    out.mkdir(parents=True)
    manifest={"schema_version":"1.0","model":a["model"],"revision":a["revision"],
              "curated_review_file_sha256":h(REVIEW),
              "numeric_manifest_sha256":h(QUANT/"manifest.json"),
              "concept_manifest_sha256":h(CONCEPT/"manifest.json"),
              "source_rights":"original physics problem generation and original editorial cards; scientific review is separately tracked",
              "data_scope":"specialist physics numerical+conceptual+non-domain selective refusal",
              "review_status":"EXPERT_REVIEWED" if a["training_permission"] else "PROVISIONAL_UNREVIEWED_RESEARCH_ONLY",
              "review_blocked_cards":a["review"]["blocked"],
              "outputs":{}}
    seen={}
    for split in ("train","valid"):
        numeric=read_checked(QUANT,"numeric",split)
        concept=read_checked(CONCEPT,"concept",split)
        rows=[]
        for row in numeric:
            rows.append({"id":"num_"+row["id"],"messages":row["messages"],"family":row["metadata"]["family"],
                        "origin":"original_programmatic_formula_verified"})
        for row in concept:
            multiplier=3 if split=="train" and row["category"]=="domain_conceptual" else (
                       2 if split=="train" and row["category"]=="out_of_scope" else 1)
            for i in range(multiplier):
                rows.append({"id":"specialist_"+row["id"]+f"_{i}","messages":row["messages"],
                             "family":row["family"],"origin":"independently_reviewed_editorial" if a["training_permission"] else "provisional_unreviewed_editorial",
                             "review_sha256":manifest["curated_review_file_sha256"]})
        random.Random("firm-4b-v1-"+split).shuffle(rows)
        for row in rows:
            prompt=row["messages"][-2]["content"]
            if prompt in heldout:raise ValueError("Evaluation prompt contamination: "+prompt)
            if prompt in seen and seen[prompt]!=split:raise ValueError("Train/valid prompt overlap")
            seen[prompt]=split
        file=out/(split+".jsonl")
        file.write_text("".join(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n" for row in rows))
        manifest["outputs"][split]={"rows":len(rows),"sha256":h(file)}
    if len(seen)<2904:raise ValueError("Unexpectedly few unique instruction prompts")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"data/processed/firm3_specialist_4b_mix_v1")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--provisional-research",action="store_true",help="Explicitly allow unreviewed cards for non-production scientific experiments")
    args=parser.parse_args()
    print(json.dumps(summary() if args.dry_run else build(args.out,provisional=args.provisional_research),indent=2))
