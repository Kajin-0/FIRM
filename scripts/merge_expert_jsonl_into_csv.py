#!/usr/bin/env python3
"""Merge expert FIRM JSONL batches into the root CSV dataset.

This script converts records of the form:

    {
      "messages": [system, user, assistant],
      "metadata": {
        "topic": "...",
        "subtopic": "...",
        "tags": "...",
        "difficulty": "...",
        "format": "..."
      }
    }

into the legacy CSV columns:

    input, output, topic, subtopic, tags, difficulty, format

Usage:
    python scripts/merge_expert_jsonl_into_csv.py \
        --csv "FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv" \
        --jsonl data/processed/firm_v2_expert_hgcdte_deep_batch01.jsonl \
        --out "FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv"

The merge deduplicates by exact input/output pair and preserves the original CSV shape.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

COLUMNS = ["input", "output", "topic", "subtopic", "tags", "difficulty", "format"]


def read_legacy_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header_row = None
    for idx, row in raw.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        if "input" in values and "output" in values:
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not find CSV header row containing input/output.")
    columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = columns
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "")].copy()
    return df[COLUMNS]


def extract_record(obj: Dict) -> Dict[str, str]:
    messages = obj.get("messages", [])
    user = ""
    assistant = ""
    for msg in messages:
        if msg.get("role") == "user":
            user = str(msg.get("content", "")).strip()
        elif msg.get("role") == "assistant":
            assistant = str(msg.get("content", "")).strip()
    meta = obj.get("metadata", {}) or {}
    return {
        "input": user,
        "output": assistant,
        "topic": str(meta.get("topic", "Expert")).strip(),
        "subtopic": str(meta.get("subtopic", "Expert Expansion")).strip(),
        "tags": str(meta.get("tags", "FIRM,expert,HgCdTe,infrared photodetectors")).strip(),
        "difficulty": str(meta.get("difficulty", "Expert")).strip(),
        "format": str(meta.get("format", "Analytical")).strip(),
    }


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = extract_record(obj)
            if not rec["input"] or not rec["output"]:
                raise ValueError(f"Missing user or assistant content at {path}:{line_no}")
            rows.append(rec)
    return pd.DataFrame(rows, columns=COLUMNS)


def write_legacy_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "", "", "", "", "", ""])
        writer.writerow(COLUMNS)
        for _, row in df.iterrows():
            writer.writerow([row.get(c, "") for c in COLUMNS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    base = read_legacy_csv(args.csv)
    add = read_jsonl(args.jsonl)
    merged = pd.concat([base, add], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["input", "output"], keep="first").reset_index(drop=True)
    removed = before - len(merged)
    write_legacy_csv(merged, args.out)

    print(f"base_rows={len(base)}")
    print(f"added_rows={len(add)}")
    print(f"duplicates_removed={removed}")
    print(f"merged_rows={len(merged)}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
