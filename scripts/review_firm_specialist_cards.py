#!/usr/bin/env python3
"""Create immutable source/claim review ledger and, after approval, a gated specialist SFT mix.

By design every new science card begins blocked. The reviewer must inspect the
card's physics claims, cite evidence, and record an explicit approval. The file
does not accept a bare boolean without source and identity. Current task only
prepares the ledger; it does NOT simulate expert approval.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
from build_firm_specialist_curriculum import DOMAIN_CARDS, OUTSIDE_TOPICS

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=ROOT/"data/reviews/firm3_specialist_concepts_v1_review.csv"
FIELDS=["kind","card_id","answer_sha256","status","reviewer","reviewed_date","evidence_url","comments"]
def create(path=TEMPLATE):
    if path.exists():raise FileExistsError(f"Refusing to overwrite review ledger: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as file:
        wr=csv.DictWriter(file,fieldnames=FIELDS,lineterminator="\n");wr.writeheader()
        for topic,q,a in DOMAIN_CARDS:
            wr.writerow({"kind":"domain_conceptual","card_id":topic,
                         "answer_sha256":hashlib.sha256(a.encode()).hexdigest(),
                         "status":"PENDING_EXPERT_REVIEW",
                         "comments":"Verify every factual/mechanistic assertion against reliable sources; check applicability/limitations."})
        for topic,q in OUTSIDE_TOPICS:
            wr.writerow({"kind":"scope_boundary","card_id":topic,
                         "answer_sha256":hashlib.sha256(
                             ("FIRM specializes in infrared photonics and related semiconductor science. "
                              "That request is outside my scope.").encode()).hexdigest(),
                         "status":"PENDING_SCOPE_REVIEW",
                         "comments":"Verify request truly unrelated; add emergency/safety exception where appropriate."})
    print(path, "cards=",len(DOMAIN_CARDS)+len(OUTSIDE_TOPICS))
def inspect(path=TEMPLATE):
    if not path.exists():raise FileNotFoundError(path)
    with path.open(encoding="utf8",newline="") as f:rows=list(csv.DictReader(f))
    cards={r["card_id"]:r for r in rows}
    if len(cards)!=len(rows):raise ValueError("Duplicate review key")
    expected={**{key:(a,"domain_conceptual") for key,q,a in DOMAIN_CARDS},
              **{key:("FIRM specializes in infrared photonics and related semiconductor science. "
                      "That request is outside my scope.","scope_boundary") for key,q in OUTSIDE_TOPICS}}
    if set(cards)!=set(expected):raise ValueError("Review ledger missing/extra cards")
    approved=set();blocked={}
    for key,(answer,kind) in expected.items():
        r=cards[key]
        if r["kind"]!=kind or hashlib.sha256(answer.encode()).hexdigest()!=r["answer_sha256"]:
            raise ValueError(f"Card integrity changed: {key}")
        acceptable=("APPROVED_EXPERT" if kind=="domain_conceptual" else "APPROVED_SCOPE")
        if r["status"]==acceptable:
            if not (r["reviewer"].strip() and r["reviewed_date"].strip() and r["evidence_url"].startswith("https://")):
                raise ValueError(f"Invalid approval metadata: {key}")
            approved.add(key)
        else:blocked[key]=r["status"]
    return {"total":len(cards),"approved":len(approved),"blocked":len(blocked),
            "approved_keys":sorted(approved),"blocked_keys":sorted(blocked)}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--create",action="store_true")
    p.add_argument("--path",type=Path,default=TEMPLATE)
    a=p.parse_args()
    if a.create:create(a.path)
    print(json.dumps(inspect(a.path),indent=2))
if __name__=="__main__":main()
