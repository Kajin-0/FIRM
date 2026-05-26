#!/usr/bin/env python3
"""Curate FIRM into a stronger expert training dataset.

Purpose
-------
The original FIRM dataset is useful but contains many repeated plug-and-chug
formula templates. This script converts the root CSV into a more robust
training corpus by:

1. reading the legacy CSV;
2. capping near-duplicate numeric templates;
3. preserving high-value HgCdTe/noise/radiometry/testing rows;
4. appending expert JSONL batches;
5. writing a curated CSV and SFT JSONL.

This does not require local manual editing. Run from repo root:

    python scripts/curate_firm_dataset_v2.py \
      --csv "FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv" \
      --expert data/processed/firm_v2_expert_hgcdte_deep_batch01.jsonl \
      --out-csv data/processed/firm_v2_curated_expert_dataset.csv \
      --out-jsonl data/processed/firm_v2_curated_expert_sft.jsonl

Design principle
----------------
A smaller, denser dataset is preferred over a larger repetitive one. FIRM should
learn expert infrared-device reasoning, not merely short equation substitution.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

COLUMNS = ["input", "output", "topic", "subtopic", "tags", "difficulty", "format"]

SYSTEM_PROMPT = (
    "You are FIRM, the Focused Infrared Research Model. Answer as a precise "
    "infrared photodetector research assistant specializing in HgCdTe/MCT, InSb, "
    "photoconductors, photodiodes, detector noise, blackbody radiometry, cryogenic "
    "testing, lock-in measurements, fabrication, and empirical troubleshooting. "
    "Use equations, units, assumptions, physical interpretation, measurement "
    "consequences, and caveats when relevant."
)

HIGH_VALUE_PAT = re.compile(
    r"HgCdTe|MCT|mercury cadmium telluride|photoconduct|Johnson|shot noise|1/f|flicker|"
    r"generation[- ]recombination|G[- ]?R|blackbody|Planck|radiometr|lock[- ]?in|MFLI|"
    r"LabOne|chopper|cryogenic|LN2|77\s*K|LPE|MOCVD|MBE|anneal|passivation|"
    r"dark current|detectivity|D\*|NEP|responsivity|cutoff|bandgap|FEniCSx|digital twin",
    re.I,
)


def normalize_template(text: str) -> str:
    text = str(text)
    text = re.sub(r"[-+]?\d*\.\d+(?:e[-+]?\d+)?", "<NUM>", text, flags=re.I)
    text = re.sub(r"[-+]?\d+(?:e[-+]?\d+)?", "<NUM>", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_legacy_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header_row = None
    for idx, row in raw.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        if "input" in values and "output" in values:
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not locate input/output header row.")
    cols = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = cols
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "")].copy()
    return df[COLUMNS].drop_duplicates(subset=["input", "output"]).reset_index(drop=True)


def read_expert_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            user = ""
            assistant = ""
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    user = str(msg.get("content", "")).strip()
                elif msg.get("role") == "assistant":
                    assistant = str(msg.get("content", "")).strip()
            meta = obj.get("metadata", {}) or {}
            rows.append({
                "input": user,
                "output": assistant,
                "topic": str(meta.get("topic", "Expert")).strip(),
                "subtopic": str(meta.get("subtopic", "Expert Expansion")).strip(),
                "tags": str(meta.get("tags", "FIRM,expert,HgCdTe,infrared photodetectors")).strip(),
                "difficulty": str(meta.get("difficulty", "Expert")).strip(),
                "format": str(meta.get("format", "Analytical")).strip(),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def curate_base(df: pd.DataFrame, max_template_regular: int = 3, max_template_high_value: int = 12) -> pd.DataFrame:
    tmp = df.copy()
    combined = (tmp["input"] + "\n" + tmp["output"] + "\n" + tmp["tags"]).astype(str)
    tmp["is_high_value"] = combined.str.contains(HIGH_VALUE_PAT, regex=True, na=False)
    tmp["template"] = tmp["input"].map(normalize_template)

    selected = []
    for _, group in tmp.groupby("template", sort=False):
        hv = group[group["is_high_value"]]
        reg = group[~group["is_high_value"]]
        if len(hv):
            selected.append(hv.head(max_template_high_value))
        if len(reg):
            selected.append(reg.head(max_template_regular))
    out = pd.concat(selected, ignore_index=True) if selected else tmp

    # Keep the dataset expert-leaning, but do not delete beginner examples entirely.
    # Cap large generic Physics groups after deduplication.
    capped = []
    for topic, group in out.groupby("topic", sort=False):
        limit = 600 if str(topic).lower() == "physics" else 250
        if len(group) > limit:
            high = group[group["is_high_value"]]
            low = group[~group["is_high_value"]]
            remaining = max(0, limit - len(high))
            if len(low) > remaining:
                low = low.sample(remaining, random_state=42)
            group = pd.concat([high, low], ignore_index=True)
        capped.append(group)
    out = pd.concat(capped, ignore_index=True)
    return out.drop(columns=["is_high_value", "template"], errors="ignore").drop_duplicates(subset=["input", "output"])


def to_sft(row: pd.Series) -> Dict:
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
        writer = csv.writer(f)
        writer.writerow(["", "", "", "", "", "", ""])
        writer.writerow(COLUMNS)
        for _, row in df.iterrows():
            writer.writerow([row.get(c, "") for c in COLUMNS])


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(to_sft(row), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--expert", action="append", type=Path, default=[])
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    args = ap.parse_args()

    base = read_legacy_csv(args.csv)
    curated = curate_base(base)

    expert_frames = [read_expert_jsonl(p) for p in args.expert]
    if expert_frames:
        curated = pd.concat([curated, *expert_frames], ignore_index=True)

    curated = curated.drop_duplicates(subset=["input", "output"], keep="first").reset_index(drop=True)

    write_csv(curated, args.out_csv)
    write_jsonl(curated, args.out_jsonl)

    print(f"base_rows={len(base)}")
    print(f"curated_rows={len(curated)}")
    print(f"wrote_csv={args.out_csv}")
    print(f"wrote_jsonl={args.out_jsonl}")


if __name__ == "__main__":
    main()
