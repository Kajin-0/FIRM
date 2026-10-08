#!/usr/bin/env python3
"""Build FIRM-4B v2.3.1 natural-prompt depth calibration corpus."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import build_firm4b_v2_3 as v23
from firm4b_v2_3_depth import DEPTH_CASES, SOURCES, word_count
from firm4b_v2_3_1_calibration import CRITICAL_PROMPTS, REASON_HINTS, MULTITURN_CASES, generic_variants

SYSTEM=(
"You are FIRM, the Focused Infrared Research Model: a compact specialist in infrared "
"photonics, semiconductor detector materials/devices, fabrication, packaging, "
"characterization, radiometry, noise and supporting physics. Short user wording does not "
"imply a short answer. By default, give a self-contained specialist answer with the context "
"needed to use the information correctly. A simple material or device identity question "
"typically deserves several informative sentences (often about 60–120 words). Comparisons, "
"engineering recommendations and diagnostics typically deserve a structured 120–300 word "
"answer or a compact table plus explanation when that is clearer. A one-sentence answer is "
"appropriate only when it genuinely resolves a narrow question. Cover decision-relevant "
"physics, tradeoffs and practical consequences without filler. When a follow-up asks for "
"more, add new technical dimensions rather than repeating the prior answer. Keep internal "
"reasoning brief even when the final answer is thorough. State assumptions, equations and "
"units when relevant. Correct false premises explicitly. Distinguish established facts from "
"inference and uncertainty. Do not invent measurements, citations, organizations, products, "
"materials or mechanisms. Decline wholly unrelated requests."
)

DEPTH_MAP={name:(q,a,src) for name,q,a,src in DEPTH_CASES}

def _rewrite_system(rows):
    out=[]
    for r in rows:
        x=json.loads(json.dumps(r))
        x["messages"][0]["content"]=SYSTEM
        out.append(x)
    return out

def _stable_subset(rows,n):
    return sorted(rows,key=lambda r:hashlib.sha256(r["id"].encode()).hexdigest())[:n]

def _inherited_rows():
    raw=_rewrite_system(v23._v22_rows()+v23._depth_rows()+v23._policy_rows()+
                        v23._precision_rows()+v23._thinking_depth_rows())
    out=[]; quant=[]
    for r in raw:
        if r["category"]=="canonical_fact":
            if int(r.get("metadata",{}).get("variant",99)) < 10:
                out.append(r)
        elif r["category"]=="quantitative_open":
            quant.append(r)
        else:
            out.append(r)
    qgroups={}
    for r in quant:
        fam=r.get("metadata",{}).get("family","NONE")
        qgroups.setdefault(fam,[]).append(r)
    for fam,rows in qgroups.items():
        limit=300 if fam=="NONE" else 200 if fam=="hgcdte_hansen" else 50
        out.extend(_stable_subset(rows,min(limit,len(rows))))
    return out

def _surface_variants(p):
    p=p.strip()
    vals=[p,p.lower(),p.rstrip("?"),"Can you explain "+p[:1].lower()+p[1:]]
    out=[]
    for x in vals:
        if x not in out: out.append(x)
    return out

def _natural_depth_rows():
    rows=[]
    for name,(base_q,a,src) in DEPTH_MAP.items():
        prompts=CRITICAL_PROMPTS.get(name)
        if prompts:
            prompts=[x for p in prompts for x in _surface_variants(p)]
        else:
            prompts=[x for p in generic_variants(base_q) for x in _surface_variants(p)]
        # Stable de-dup.
        seen=set(); prompts=[p for p in prompts if not (p in seen or seen.add(p))]
        for i,prompt in enumerate(prompts):
            rows.append({"id":f"v231_nat_{name}_{i:03d}","category":"natural_depth",
              "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research_claim_checked","rights":"original",
                          "evidence_url":SOURCES[src],"target_words":word_count(a),"natural_prompt":True}})
    return rows

def _thinking_rows():
    rows=[]
    for name,(base_q,a,src) in DEPTH_MAP.items():
        prompts=CRITICAL_PROMPTS.get(name, generic_variants(base_q))
        prompts=prompts[:10]
        hint=REASON_HINTS[name]
        for i,prompt in enumerate(prompts):
            rows.append({"id":f"v231_think_{name}_{i:02d}","category":"natural_depth_thinking",
              "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt},
                          {"role":"assistant","reasoning_content":hint,"content":a}],
              "metadata":{"review_status":"provisional_reasoning_trace","rights":"original",
                          "evidence_url":SOURCES[src],"thinking_trace":True,
                          "target_words":word_count(a),"natural_prompt":True}})
    return rows

FOLLOWUPS=[
    "{q}",
    "{ql}",
    "Can you go deeper on that?",
    "What else matters in practice?",
    "That seems incomplete. What should I know?",
    "Expand that into something useful for an engineer.",
    "What are the important details you left out?",
    "How does that affect the actual detector or package design?",
    "Can you unpack that further?",
    "What does that mean for the design?",
    "Can you explain the engineering implications?",
    "What would you check in practice?",
    "Give me the fuller picture.",
    "What am I missing?",
    "How does this matter in real hardware?",
    "Walk through the relevant tradeoffs.",
]

def _multiturn_rows():
    rows=[]
    for name,u1,a1,u2,target in MULTITURN_CASES:
        _,a,src=DEPTH_MAP[target]
        q0=u2.rstrip("?"); ql=q0[:1].lower()+q0[1:]
        for i,w in enumerate(FOLLOWUPS):
            follow=w.format(q=u2,ql=ql)
            rows.append({"id":f"v231_multi_{name}_{i:02d}","category":"multiturn_depth",
              "messages":[{"role":"system","content":SYSTEM},
                          {"role":"user","content":u1},
                          {"role":"assistant","content":a1},
                          {"role":"user","content":follow},
                          {"role":"assistant","content":a}],
              "metadata":{"review_status":"provisional_research_claim_checked","rights":"original",
                          "evidence_url":SOURCES[src],"target_words":word_count(a),
                          "multiturn":True}})
    return rows

def _stats(rows):
    vals=sorted(word_count(r["messages"][-1].get("content","")) for r in rows)
    def pct(p): return vals[round((len(vals)-1)*p)]
    return {"mean_words":round(sum(vals)/len(vals),2),"median_words":pct(.5),
            "p75_words":pct(.75),"p90_words":pct(.9),"p95_words":pct(.95),"max_words":max(vals)}

def build(out:Path):
    if out.exists(): raise FileExistsError(out)
    raw=_inherited_rows()+_natural_depth_rows()+_thinking_rows()+_multiturn_rows()
    # Deduplicate on the actual generation input: masked conversation prefix plus
    # thinking mode. Prefer focused calibration rows when an inherited prompt collides.
    priority={"multiturn_depth":5,"natural_depth_thinking":5,"natural_depth":4,
              "reasoning_depth":3,"depth_complete":3,"precision_correction":3}
    uniq={}
    for r in raw:
        a=r["messages"][-1]
        thinking=isinstance(a.get("reasoning_content"),str) and bool(a.get("reasoning_content","").strip())
        k=json.dumps({"thinking":thinking,"prefix":r["messages"][:-1]},ensure_ascii=False,sort_keys=True)
        if k not in uniq or priority.get(r["category"],1)>priority.get(uniq[k]["category"],1):
            uniq[k]=r
    allrows=list(uniq.values())
    rng=random.Random(23120261008); rng.shuffle(allrows)
    cats={}
    for r in allrows: cats.setdefault(r["category"],[]).append(r)
    train=[];valid=[]
    for cat,rows in cats.items():
        k=max(2,round(len(rows)*.08))
        valid+=rows[:k];train+=rows[k:]
    rng.shuffle(train);rng.shuffle(valid)
    out.mkdir(parents=True)
    for split,rows in (("train",train),("valid",valid)):
        (out/(split+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows))
    calib=[r for r in allrows if r["category"] in {"natural_depth","natural_depth_thinking","multiturn_depth"}]
    manifest={
      "schema_version":"2.3.1",
      "rows":{"train":len(train),"valid":len(valid)},
      "categories":{c:len(v) for c,v in sorted(cats.items())},
      "inherited_from":"firm4b_v2_3_expanded_v1",
      "calibration_rows":len(calib),
      "calibration_fraction":round(len(calib)/len(allrows),4),
      "answer_stats":_stats(allrows),
      "calibration_answer_stats":_stats(calib),
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
    ap.add_argument("--out",type=Path,default=Path("data/processed/firm4b_v2_3_1_expanded_v1"))
    build(ap.parse_args().out)
