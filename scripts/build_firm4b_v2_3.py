#!/usr/bin/env python3
"""FIRM-4B v2.3 curriculum: adaptive answer depth + precision repair.

Builds on the complete v2.2 curriculum, rewrites the system policy toward
adaptive completeness, and adds long-form technical targets for comparisons,
packaging, optical materials, detector physics, diagnostics, and process
engineering. Exact user stress prompts remain held out.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import build_firm4b_v2_2 as v22
from firm4b_v2_3_depth import DEPTH_CASES, DEPTH_POLICY_CASES, PRECISION_CORRECTIONS, SOURCES, word_count
from firm4b_v2_3_reasoning import THINKING_DEPTH_CASES

SYSTEM=(
"You are FIRM, the Focused Infrared Research Model: a compact specialist in infrared "
"photonics, semiconductor detector materials/devices, fabrication, packaging, "
"characterization, radiometry, noise and supporting physics. Match answer depth to the "
"technical task. A simple identity question should still establish the material or device, "
"its defining physical property, and why it matters. Comparisons should cover the "
"decision-relevant dimensions rather than stopping after taxonomy. Engineering and "
"diagnostic questions should give concrete options, governing tradeoffs, likely failure "
"modes, and practical validation steps. Be concise in wording but not incomplete in "
"technical content. Expand materially when the user asks for more detail. State assumptions, "
"equations and units when relevant. Correct false premises explicitly. Distinguish established "
"facts from inference and uncertainty. Do not invent measurements, citations, organizations, "
"products, materials or mechanisms. Decline wholly unrelated requests."
)

DEPTH_WRAPPERS=[
    "{q}",
    "Give a rigorous but compact technical explanation: {q}",
    "Answer with enough detail to support an engineering decision: {q}",
    "Explain this for an infrared detector engineer: {q}",
    "Give the complete technical picture without filler: {q}",
    "For a detector design review, {q}",
    "Include the relevant physics and practical implications: {q}",
    "Answer as a technical reference note: {q}",
    "What should an experienced IR engineer know about this? {q}",
    "Do not stop at taxonomy; explain the decision-relevant differences: {q}",
    "For a lab or process engineer, explain: {q}",
    "Give the material/device context, tradeoffs, and practical consequence: {q}",
    "Explain this well enough that someone could use the answer in real detector work: {q}",
    "Technical deep dive, but stay focused: {q}",
    "Provide a complete specialist answer: {q}",
    "In practical infrared engineering, {q}",
]

POLICY_WRAPPERS=[
    "{q}","Answer as a model-response design rule: {q}","For FIRM response quality, {q}",
    "What is the preferred specialist behavior here? {q}","Explain the response standard: {q}",
    "For a compact expert model, {q}","What should a strong technical assistant do? {q}",
    "State the adaptive-depth rule: {q}",
]

def _rewrite_system(rows):
    out=[]
    for r in rows:
        x=json.loads(json.dumps(r))
        x["messages"][0]["content"]=SYSTEM
        out.append(x)
    return out

def _v22_rows():
    inherited=(v22.v21._extra_fact_rows()+v22.v21._extra_taxonomy_rows()+v22.v21.base.quantitative_rows()+
               v22.v21._extra_diag_rows()+v22.v21.base.refusal_rows()+v22.v21._contrastive_rows())
    return (_rewrite_system(inherited)+
            _rewrite_system(v22._rows_from_facts())+
            _rewrite_system(v22._rows_from_diag())+
            _rewrite_system(v22._rows_from_contrastive())+
            _rewrite_system(v22._rows_from_thinking()))

def _depth_rows():
    rows=[]
    for i,(_,q,a,src) in enumerate(DEPTH_CASES):
        for v,wrapper in enumerate(DEPTH_WRAPPERS):
            prompt=wrapper.format(q=v22.v21.base.perturb(q,v%12))
            rows.append({"id":f"v23_depth_{i}_{v}","category":"depth_complete",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":prompt},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research_claim_checked",
                          "rights":"original","evidence_url":SOURCES[src],
                          "target_words":word_count(a)}})
    return rows

def _policy_rows():
    rows=[]
    for i,(q,a) in enumerate(DEPTH_POLICY_CASES):
        for v,wrapper in enumerate(POLICY_WRAPPERS):
            prompt=wrapper.format(q=v22.v21.base.perturb(q,v%12))
            rows.append({"id":f"v23_policy_{i}_{v}","category":"depth_policy",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":prompt},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_style_policy","rights":"original",
                          "target_words":word_count(a)}})
    return rows

def _precision_rows():
    rows=[]
    for i,(q,a) in enumerate(PRECISION_CORRECTIONS):
        for v,wrapper in enumerate(POLICY_WRAPPERS):
            prompt=wrapper.format(q=v22.v21.base.perturb(q,v%12))
            rows.append({"id":f"v23_precision_{i}_{v}","category":"precision_correction",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":prompt},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research_claim_checked","rights":"original"}})
    return rows

def _thinking_depth_rows():
    rows=[]
    for i,(q,reasoning,a) in enumerate(THINKING_DEPTH_CASES):
        for v,wrapper in enumerate(POLICY_WRAPPERS):
            prompt=wrapper.format(q=v22.v21.base.perturb(q,v%12))
            rows.append({"id":f"v23_thinkdepth_{i}_{v}","category":"reasoning_depth",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":prompt},
                          {"role":"assistant","reasoning_content":reasoning,"content":a}],
              "metadata":{"review_status":"provisional_reasoning_trace",
                          "rights":"original","thinking_trace":True,
                          "target_words":word_count(a)}})
    return rows

def _answer_stats(rows):
    lens=sorted(word_count(r["messages"][-1].get("content","")) for r in rows)
    def pct(p):
        return lens[round((len(lens)-1)*p)]
    return {
      "mean_words":round(sum(lens)/len(lens),2),
      "median_words":pct(.5),
      "p75_words":pct(.75),
      "p90_words":pct(.9),
      "p95_words":pct(.95),
      "max_words":max(lens),
    }

def build(out:Path):
    if out.exists(): raise FileExistsError(out)
    raw=_v22_rows()+_depth_rows()+_policy_rows()+_precision_rows()+_thinking_depth_rows()
    by_prompt={}
    for row in raw:
        by_prompt.setdefault(row["messages"][1]["content"],row)
    allrows=list(by_prompt.values())
    rng=random.Random(23032026);rng.shuffle(allrows)
    cats={}
    for x in allrows:cats.setdefault(x["category"],[]).append(x)
    train=[];valid=[]
    for cat,rows in cats.items():
        k=max(2,round(len(rows)*.08))
        valid+=rows[:k];train+=rows[k:]
    rng.shuffle(train);rng.shuffle(valid)
    out.mkdir(parents=True)
    for split,rows in (("train",train),("valid",valid)):
        (out/(split+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows))
    depth=[r for r in allrows if r["category"] in {"depth_complete","depth_policy","reasoning_depth"}]
    manifest={
      "schema_version":"2.3",
      "rows":{"train":len(train),"valid":len(valid)},
      "categories":{c:sum(1 for r in allrows if r["category"]==c) for c in sorted(cats)},
      "inherited_from":"firm4b_v2_2_expanded_v1",
      "depth_case_targets":len(DEPTH_CASES),
      "depth_policy_targets":len(DEPTH_POLICY_CASES),
      "precision_correction_targets":len(PRECISION_CORRECTIONS),
      "reasoning_depth_targets":len(THINKING_DEPTH_CASES),
      "answer_stats":_answer_stats(allrows),
      "depth_answer_stats":_answer_stats(depth),
      "source_urls":sorted(set(v22.v21.base.SOURCES.values())|set(v22.v21.EXTRA_SOURCES.values())|
                           set(v22.SOURCES.values())|set(SOURCES.values())),
      "private_benchmark_imported":False,
      "exact_user_stress_prompts_imported":False,
      "status":"PROVISIONAL_RESEARCH_CLAIM_CHECKED_NOT_EXPERT_RELEASE_APPROVED",
      "train_sha256":hashlib.sha256((out/"train.jsonl").read_bytes()).hexdigest(),
      "valid_sha256":hashlib.sha256((out/"valid.jsonl").read_bytes()).hexdigest(),
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2))
    return manifest

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("data/processed/firm4b_v2_3_expanded_v1"))
    build(ap.parse_args().out)
