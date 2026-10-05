#!/usr/bin/env python3
"""Inventory and audit every legacy training source, including rewrite replacements.

No data is modified. Reports use source hashes; optional tokenizer never loads weights.
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from firm_data import audit_records, leakage, load_evals, read_records, sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("data/audits/firm3_2026-10-05"))
    ap.add_argument("--date", default="2026-10-05")
    ap.add_argument("--tokenizer")
    ap.add_argument("--tokenizer-revision")
    ap.add_argument("--tokenizer-files", type=Path, help="Already saved tokenizer directory for genuinely offline audits")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    tokenizer = None
    if args.tokenizer:
        if not args.tokenizer_revision:
            ap.error("--tokenizer-revision is required for reproducibility")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_files or args.tokenizer), revision=args.tokenizer_revision,
                                                  local_files_only=args.local_files_only,
                                                  trust_remote_code=False)
    root = args.root
    files = sorted({*root.glob("*.csv"), *root.glob("data/processed/*.csv"),
                    *root.glob("data/processed/*.jsonl"), *root.glob("data/curation/**/*.jsonl")})
    eval_items = load_evals(root / "evals")
    report = {"schema_version": "1.0", "date": args.date,
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "tool_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("firm_data.py")]},
              "tokenizer": {"model": args.tokenizer, "revision": args.tokenizer_revision,
                            "files": {p.name: sha256(p) for p in sorted(args.tokenizer_files.glob("*")) if p.is_file()}
                                     if args.tokenizer_files else None,
                            "status": "measured" if tokenizer else "not measured; no intended tokenizer installed"},
              "sources": [], "eval_files": [{"path": str(p.relative_to(root)), "sha256": sha256(p),
                  "records": len(read_records(p, evaluation=True)[0])} for p in sorted((root / "evals").glob("*.jsonl"))]}
    for path in files:
        rows, parsing = read_records(path)
        role = "rewrite_transform" if "curation" in path.parts else "training_candidate"
        source = {"path": str(path.relative_to(root)), "role": role, "bytes": path.stat().st_size,
                  "sha256": sha256(path), "records": len(rows), "parsing": parsing,
                  "metrics": audit_records(rows, tokenizer)}
        report["sources"].append(source)
        leak = leakage(rows, eval_items)
        write_json(args.out / (path.stem + "_leakage.json"), {"source": source["path"],
                   "source_sha256": source["sha256"], **leak})
        source["leakage_candidate_pairs"] = len(leak["findings"])
        print(f"{source['path']}: {len(rows)} records, {len(leak['findings'])} leakage candidates", flush=True)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    tracked = [p for p in tracked if p]
    inventory = {"tracked_files": len(tracked), "by_extension": dict(Counter(Path(p).suffix or "<none>" for p in tracked)),
                 "by_top_level": dict(Counter(Path(p).parts[0] if len(Path(p).parts) > 1 else "<root>" for p in tracked)),
                 "files": [{"path": p, "bytes": (root / p).stat().st_size,
                            "lines": len((root / p).read_bytes().splitlines())} for p in tracked],
                 "python_loc_including_comments": sum(len((root / p).read_text().splitlines()) for p in tracked if p.endswith(".py"))}
    # Keep the original takeover inventory; subsequent snapshots have their own identity.
    if not (args.out / "inventory.json").exists():
        write_json(args.out / "inventory.json", inventory)
    else:
        write_json(args.out / "inventory_current.json", inventory)
    write_json(args.out / "dataset_audit.json", report)
    lines = ["# FIRM 3 reproducible dataset snapshot", "", f"Date: {args.date}; source HEAD: `{report['git_sha']}`.",
             "", "Sources are alternatives/transformations, not additive independent training records.",
             "Heuristic feature counts indicate mentions, not validated scientific capability.", "",
             "| Source | Records | Mean prompt words | Mean response words | Response p95/max | Duplicate responses | Leakage candidates |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in report["sources"]:
        m = s["metrics"]
        if m["response_words"]["count"]:
            lines.append(f"| {s['path']} | {s['records']} | {m['prompt_words']['mean']:.2f} | {m['response_words']['mean']:.2f} | "
                         f"{m['response_words']['p95']:.1f}/{m['response_words']['max']} | "
                         f"{m['duplicates']['exact']['responses']['extra_records']} | {s['leakage_candidate_pairs']} |")
    lines += ["", "Full statistics, parsing diagnostics, distributions, template families and near-pair counts are in `dataset_audit.json`.",
              "Per-source leakage files contain review candidates and nearest neighbors, not contamination verdicts.",
              "Tokenization: " + json.dumps(report["tokenizer"]), ""]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
