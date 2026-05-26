#!/usr/bin/env python3
"""Prepare the FIRM dataset for supervised fine-tuning.

Input: CSV with columns resembling:
    input, output, topic, subtopic, tags, difficulty, format

Outputs:
    data/processed/firm_v0_full_sft_messages.jsonl
    data/processed/firm_v1_balanced_sft_messages.jsonl
    data/processed/firm_v1_balanced_train.jsonl
    data/processed/firm_v1_balanced_valid.jsonl
    data/processed/firm_v1_balanced_test.jsonl
    data/audits/topic_distribution.csv
    data/audits/keyword_coverage.csv
    data/audits/top_template_families.csv
    data/audits/firm_dataset_flags_review.csv

Usage:
    python scripts/prepare_firm_dataset.py \
        --input data/raw/FIRM_Dataset_Sheet1_updated.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ["input", "output", "topic", "subtopic", "tags", "difficulty", "format"]

SYSTEM_PROMPT = (
    "You are FIRM, the Focused Infrared Research Model. "
    "You are a precise infrared photodetector research assistant specializing in HgCdTe/MCT, "
    "InSb, photoconductors, photodiodes, detector noise, radiometry, lock-in measurements, "
    "cryogenic testing, fabrication, passivation, and empirical troubleshooting. "
    "When relevant, answer with assumptions, equations, units, physical interpretation, "
    "measurement consequences, and caveats."
)

KEYWORD_PATTERNS: Dict[str, str] = {
    "HgCdTe/MCT": r"\b(HgCdTe|MCT|mercury cadmium telluride)\b",
    "InSb": r"\bInSb\b",
    "InAsSb": r"\bInAsSb\b",
    "T2SL": r"\b(T2SL|type[- ]II superlattice|superlattice)\b",
    "QWIP": r"\bQWIP\b",
    "Photoconductor": r"\bphotoconduct(or|ive|ivity)\b",
    "Photodiode": r"\bphotodiode\b",
    "Responsivity": r"\bresponsivit(y|ies)\b|\bV/W\b|\bA/W\b",
    "Detectivity/D*": r"\bdetectivit(y|ies)\b|\bD\*\b",
    "Noise": r"\bnoise\b|\bPSD\b|\bNEP\b",
    "Johnson noise": r"\bJohnson\b|\bthermal noise\b",
    "1/f noise": r"\b1/f\b|\bflicker\b",
    "G-R noise": r"\bG[- ]?R\b|generation[- ]recombination",
    "Shot noise": r"\bshot noise\b",
    "Blackbody": r"\bblackbody\b|\bPlanck\b",
    "Lock-in/MFLI": r"\block[- ]?in\b|\bMFLI\b|\bLabOne\b",
    "Chopper": r"\bchopper\b|\bchopped\b",
    "Cryogenic/LN2": r"\bcryogenic\b|\bLN2\b|liquid nitrogen|\b77\s*K\b",
    "LPE": r"\bLPE\b|liquid phase epitaxy",
    "MOCVD": r"\bMOCVD\b",
    "MBE": r"\bMBE\b|molecular beam epitaxy",
    "Anneal": r"\banneal",
    "Passivation": r"\bpassivation\b|\bCdTe\b|\bZnS\b",
    "Readout/ROIC": r"\bROIC\b|\breadout\b",
    "FPA": r"\bFPA\b|focal plane array",
    "FEniCSx/Digital twin": r"\bFEniCSx\b|\bdigital twin\b|\bDOLFINx\b",
    "Radiometry": r"\bradiometr|radiance|irradiance|etendue|étendue",
    "Bandgap/cutoff": r"\bband ?gap\b|\bE_g\b|cutoff|\blambda_c\b|\bλ_c\b",
}

UNIT_PATTERN = re.compile(
    r"\b(eV|K|Hz|A/W|V/W|W|V|A|Ohm|Ω|cm\^-?3|m\^-?2|m\^2|cm\^2|nm|um|µm|mm|s|ms|us|µs)\b"
)
EQUATION_PATTERN = re.compile(r"[=∝≈]|\bexp\(|\bsqrt\(|\bD\*\b|\bNEP\b|\bR_V\b|\bE_g\b")


def read_firm_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)

    header_row = None
    for idx, row in raw.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        if "input" in values and "output" in values:
            header_row = idx
            break

    if header_row is None:
        raise ValueError("Could not locate a header row containing input/output columns.")

    columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = columns
    df = df[[c for c in REQUIRED_COLUMNS if c in df.columns]].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    df = df[(df["input"] != "") & (df["output"] != "")].copy()
    df = df.drop_duplicates(subset=["input", "output"]).reset_index(drop=True)
    return df


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def normalize_template(text: str) -> str:
    text = str(text)
    text = re.sub(r"[-+]?\d*\.\d+(?:e[-+]?\d+)?", "<NUM>", text, flags=re.I)
    text = re.sub(r"[-+]?\d+(?:e[-+]?\d+)?", "<NUM>", text, flags=re.I)
    return text.strip()


def to_message(row: pd.Series) -> dict:
    metadata = {
        "topic": row.get("topic", ""),
        "subtopic": row.get("subtopic", ""),
        "tags": row.get("tags", ""),
        "difficulty": row.get("difficulty", ""),
        "format": row.get("format", ""),
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["input"]).strip()},
            {"role": "assistant", "content": str(row["output"]).strip()},
        ],
        "metadata": metadata,
    }


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_audits(df: pd.DataFrame, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    tmp["input_words"] = tmp["input"].map(word_count)
    tmp["output_words"] = tmp["output"].map(word_count)
    tmp["has_equation_like_output"] = tmp["output"].map(lambda x: bool(EQUATION_PATTERN.search(str(x))))
    tmp["has_unit_like_output"] = tmp["output"].map(lambda x: bool(UNIT_PATTERN.search(str(x))))
    tmp["input_template"] = tmp["input"].map(normalize_template)

    tmp.to_csv(audit_dir / "firm_dataset_flags_review.csv", index=False)

    topic_dist = tmp["topic"].value_counts(dropna=False).rename_axis("topic").reset_index(name="count")
    topic_dist["percent"] = 100.0 * topic_dist["count"] / len(tmp)
    topic_dist.to_csv(audit_dir / "topic_distribution.csv", index=False)

    coverage = []
    joined = (tmp["input"] + "\n" + tmp["output"] + "\n" + tmp["tags"]).astype(str)
    for name, pat in KEYWORD_PATTERNS.items():
        mask = joined.str.contains(pat, case=False, regex=True, na=False)
        coverage.append({"keyword_area": name, "count": int(mask.sum()), "percent": float(100.0 * mask.sum() / len(tmp))})
    pd.DataFrame(coverage).to_csv(audit_dir / "keyword_coverage.csv", index=False)

    fam = tmp.groupby("input_template", dropna=False).size().reset_index(name="count")
    fam = fam.sort_values("count", ascending=False)
    examples = tmp.drop_duplicates("input_template")[["input_template", "input"]]
    fam = fam.merge(examples, on="input_template", how="left").rename(columns={"input": "example_input"})
    fam.to_csv(audit_dir / "top_template_families.csv", index=False)


def make_balanced(df: pd.DataFrame, max_per_template: int = 3, max_per_topic: int = 200) -> pd.DataFrame:
    tmp = df.copy()
    tmp["input_template"] = tmp["input"].map(normalize_template)
    balanced = []
    for _, group in tmp.groupby("input_template", sort=False):
        balanced.append(group.head(max_per_template))
    out = pd.concat(balanced, ignore_index=True)

    capped = []
    for _, group in out.groupby("topic", sort=False):
        if len(group) > max_per_topic:
            capped.append(group.sample(max_per_topic, random_state=42))
        else:
            capped.append(group)
    return pd.concat(capped, ignore_index=True).drop(columns=["input_template"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--processed-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--audit-dir", default=Path("data/audits"), type=Path)
    args = parser.parse_args()

    df = read_firm_csv(args.input)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    build_audits(df, args.audit_dir)

    full_messages = [to_message(row) for _, row in df.iterrows()]
    write_jsonl(full_messages, args.processed_dir / "firm_v0_full_sft_messages.jsonl")

    balanced = make_balanced(df)
    balanced_messages = [to_message(row) for _, row in balanced.iterrows()]
    write_jsonl(balanced_messages, args.processed_dir / "firm_v1_balanced_sft_messages.jsonl")

    train_valid, test = train_test_split(balanced, test_size=0.05, random_state=42, shuffle=True)
    train, valid = train_test_split(train_valid, test_size=0.0526, random_state=42, shuffle=True)

    write_jsonl([to_message(row) for _, row in train.iterrows()], args.processed_dir / "firm_v1_balanced_train.jsonl")
    write_jsonl([to_message(row) for _, row in valid.iterrows()], args.processed_dir / "firm_v1_balanced_valid.jsonl")
    write_jsonl([to_message(row) for _, row in test.iterrows()], args.processed_dir / "firm_v1_balanced_test.jsonl")

    print(f"Loaded rows: {len(df)}")
    print(f"Balanced rows: {len(balanced)}")
    print(f"Train/valid/test: {len(train)}/{len(valid)}/{len(test)}")


if __name__ == "__main__":
    main()
