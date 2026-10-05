#!/usr/bin/env python3
"""Record hashes/provenance for unchanged legacy eval assets. Never overwrite a manifest."""
import argparse
import subprocess
from pathlib import Path

from firm_data import digest, require_clean, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=Path("evals"))
    ap.add_argument("--out", type=Path, default=Path("evals/firm_legacy_v1_manifest.json"))
    ap.add_argument("--date", default="2026-10-05")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Manifest exists; preserve it and create a new version")
    assets = []
    ids = set()
    for path in sorted(args.eval_dir.glob("*.jsonl")):
        rows = require_clean(path, evaluation=True)
        records = []
        for row in rows:
            item = row["raw"]
            if item["id"] in ids:
                raise ValueError("Duplicate eval ID: " + item["id"])
            ids.add(item["id"])
            records.append({"id": item["id"], "record_sha256": digest(item),
                            "prompt_sha256": digest(row["input"]),
                            "category": item.get("category", item.get("class", path.stem)),
                            "grading": "human_rubric" if "expected_traits" in item else "keyword_proxy_only"})
        assets.append({"path": str(path), "sha256": sha256(path), "records": records})
    write_json(args.out, {"schema_version": "1.0", "benchmark_version": "firm_legacy_v1",
                         "generation_date": args.date,
                         "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                         "provenance": "Existing repository evals; prompts and rubrics preserved byte-for-byte",
                         "answers": "public rubrics; no hidden solutions",
                         "use": "legacy diagnostic; training overlap exists; not an uncontaminated capability benchmark",
                         "leakage_check": {"status": "candidates found", "report": "data/audits/firm3_2026-10-05",
                                           "semantic_review": "see docs/FIRM3_AUDIT.md"},
                         "assets": assets})


if __name__ == "__main__":
    main()
