#!/usr/bin/env python3
"""Apply numbered FIRM rewrite batches to the original large CSV.

This is the preferred dataset-improvement workflow:

1. Keep the broad original dataset.
2. Improve original rows in auditable rewrite batches.
3. Deduplicate exact final rows.
4. Export both CSV and SFT JSONL.

Usage:
    python scripts/apply_rewrite_batches.py \
      --base-csv "FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv" \
      --rewrite-glob "data/curation/rewrite_batches/firm_rewrite_batch_*.jsonl" \
      --out-csv data/processed/firm_rewritten_large_dataset.csv \
      --out-jsonl data/processed/firm_rewritten_large_sft.jsonl \
      --report data/audits/firm_rewrite_report.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

COLUMNS = ["input", "output", "topic", "subtopic", "tags", "difficulty", "format"]

SYSTEM_PROMPT = (
    "You are FIRM, the Focused Infrared Research Model. Answer as a precise infrared "
    "photodetector research assistant for expert engineers and scientists. Use equations, "
    "units, assumptions, empirical caveats, and measurement implications when relevant."
)


def normalize_match_text(value: object) -> str:
    """Normalize visually identical CSV/rewrite text for robust matching.

    The legacy dataset contains non-breaking spaces in some unit strings, for example
    "0.046\u00a0eV" and "302.9\u00a0K". Rewrite batches are often authored with normal
    spaces. Matching should not fail only because of this invisible whitespace variant.
    """
    return " ".join(str(value).replace("\u00a0", " ").split()).strip()


def read_legacy_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header_row = None
    for idx, row in raw.iterrows():
        vals = [str(v).strip().lower() for v in row.tolist()]
        if "input" in vals and "output" in vals:
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not find header row containing input/output")
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "")].copy()
    return df[COLUMNS].reset_index(drop=True)


def read_rewrite_batches(pattern: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(pattern)):
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                obj["_source_file"] = str(p)
                obj["_source_line"] = line_no
                rows.append(obj)
    return rows


def apply_rewrites(df: pd.DataFrame, rewrites: List[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    input_to_indices: Dict[str, List[int]] = {}
    for idx, row in out.iterrows():
        input_to_indices.setdefault(normalize_match_text(row["input"]), []).append(idx)

    report = []
    seen_targets = set()
    for rw in rewrites:
        match_original = str(rw["match_input"]).strip()
        match = normalize_match_text(match_original)
        repl = rw["replacement"]
        matches = input_to_indices.get(match, [])
        if not matches:
            status = "NO_MATCH"
        else:
            # Apply to all normalized input matches; normally one row.
            for idx in matches:
                for col in COLUMNS:
                    out.at[idx, col] = str(repl.get(col, out.at[idx, col])).strip()
            status = "APPLIED"
        target_key = (str(repl.get("input", "")).strip(), str(repl.get("output", "")).strip())
        duplicate_target = target_key in seen_targets
        seen_targets.add(target_key)
        report.append({
            "batch": rw.get("batch", ""),
            "source_file": rw.get("_source_file", ""),
            "source_line": rw.get("_source_line", ""),
            "original_row_index": rw.get("original_row_index", ""),
            "match_input": match_original,
            "normalized_match_input": match,
            "status": status,
            "match_count": len(matches),
            "replacement_input": repl.get("input", ""),
            "duplicate_target_within_batches": duplicate_target,
        })
    return out, pd.DataFrame(report)


def to_sft(row: pd.Series) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["input"]).strip()},
            {"role": "assistant", "content": str(row["output"]).strip()},
        ],
        "metadata": {c: str(row[c]).strip() for c in ["topic", "subtopic", "tags", "difficulty", "format"]},
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "", "", "", "", "", ""])
        w.writerow(COLUMNS)
        for _, row in df.iterrows():
            w.writerow([row[c] for c in COLUMNS])


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(to_sft(row), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", required=True, type=Path)
    ap.add_argument("--rewrite-glob", required=True)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    base = read_legacy_csv(args.base_csv)
    rewrites = read_rewrite_batches(args.rewrite_glob)
    rewritten, report = apply_rewrites(base, rewrites)

    before_dedup = len(rewritten)
    rewritten = rewritten.drop_duplicates(subset=["input", "output"], keep="first").reset_index(drop=True)
    after_dedup = len(rewritten)

    write_csv(rewritten, args.out_csv)
    write_jsonl(rewritten, args.out_jsonl)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    print(f"base_rows={len(base)}")
    print(f"rewrite_rows={len(rewrites)}")
    print(f"applied={(report['status'] == 'APPLIED').sum()}")
    print(f"no_match={(report['status'] == 'NO_MATCH').sum()}")
    print(f"dedup_removed={before_dedup-after_dedup}")
    print(f"final_rows={after_dedup}")
    print(f"out_csv={args.out_csv}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
