#!/usr/bin/env python3
"""Lightweight keyword-based evaluator for FIRM outputs.

This script does not generate model outputs. It scores an existing JSONL file of
model answers against an eval JSONL file.

Expected predictions JSONL:
    {"id": "v2_001", "answer": "..."}

Usage:
    python scripts/eval_firm_keywords.py \
      --eval evals/firm_v2_expert_eval.jsonl \
      --pred predictions/firm_v2_answers.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(s: str) -> str:
    return str(s).lower().replace("π", "pi").replace("μ", "u")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    args = ap.parse_args()

    eval_rows = {r["id"]: r for r in load_jsonl(args.eval)}
    pred_rows = {r["id"]: r for r in load_jsonl(args.pred)}

    total = 0
    score_sum = 0.0
    for eid, item in eval_rows.items():
        answer = norm(pred_rows.get(eid, {}).get("answer", ""))
        must = item.get("must_include", [])
        fail = item.get("failure_modes", [])
        hit = sum(1 for k in must if norm(k) in answer)
        miss = [k for k in must if norm(k) not in answer]
        bad = [k for k in fail if norm(k) in answer]
        score = hit / max(1, len(must))
        if bad:
            score *= 0.5
        total += 1
        score_sum += score
        print(f"{eid}: score={score:.3f} hits={hit}/{len(must)} bad={len(bad)}")
        if miss:
            print(f"  missing: {miss}")
        if bad:
            print(f"  failure-trigger: {bad}")

    print(f"\nmean_score={score_sum/max(1,total):.3f}")


if __name__ == "__main__":
    main()
