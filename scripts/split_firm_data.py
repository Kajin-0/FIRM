#!/usr/bin/env python3
"""Build deterministic candidate splits; quarantine benchmark matches by whole group.

Does not overwrite sources or publish a claim of scientific review. Output directory
must be new. Existing numeric-template cap is reused, now with group-level separation.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from firm_data import calculation_checks, digest, leakage, load_evals, normalize, require_clean, sha256, template, words, write_json, write_jsonl

SYSTEM = ("You are FIRM, the Focused Infrared Research Model, specializing in infrared "
          "photodetector science and engineering. State relevant assumptions, equations, "
          "variables and units, calculations, sanity checks, physical interpretation, "
          "competing hypotheses and validation measurements.")


def components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    seen = {}
    for i, row in enumerate(rows):
        keys = [("prompt_template", template(row["input"]))]
        if len(words(row["output"])) >= 8:
            keys.append(("response_template", template(row["output"])))
        for key in ("family_id", "document_id"):
            if row["metadata"].get(key):
                keys.append((key, str(row["metadata"][key])))
        keys.extend(tuple(key) for key in row.get("declared_groups", []))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def build_splits(rows, eval_items, seed=42, valid_fraction=.1, test_fraction=.1, cap=3, review_exclusions=None):
    if not 0 < valid_fraction < 1 or not 0 < test_fraction < 1 or valid_fraction + test_fraction >= 1:
        raise ValueError("Invalid split fractions")
    if cap < 1:
        raise ValueError("Template cap must be positive")
    # Sorting and case-preserving pair IDs make input ordering irrelevant.
    merged = {}
    # Preserve case, line structure and units when deduplicating. Case-folding is
    # useful for conservative leakage flags, but mW/MW and symbols are not equal.
    ordered = sorted(rows, key=lambda r: digest({"input": r["input"], "output": r["output"],
                                              "messages": r.get("messages", []), "metadata": r["metadata"]}))
    for row in ordered:
        identity = [row["input"].strip(), row["output"].strip()]
        if sum(m["role"] == "user" for m in row.get("messages", [])) > 1:
            identity.append(row["messages"])
        key = digest(identity)
        if key not in merged:
            merged[key] = {**row, "id": "firm3_" + key, "origins": [], "declared_groups": []}
        merged[key]["origins"].extend(row.get("origins", []))
        for name in ("family_id", "document_id"):
            if row["metadata"].get(name):
                merged[key]["declared_groups"].append((name, str(row["metadata"][name])))
    candidates = sorted(merged.values(), key=lambda r: r["id"])
    for i, row in enumerate(candidates):
        row["line"] = i + 1  # Local candidate index, not source line.
        row["origins"] = sorted(row["origins"], key=lambda o: (o["path"], o["line"]))
        row["declared_groups"] = sorted(set(row["declared_groups"]))
    detected = leakage(candidates, eval_items)
    excluded = {f["source_line"] for f in detected["findings"]}
    scientific = calculation_checks(candidates)
    excluded.update(r["line"] for r in scientific["flagged"])
    detected["scientific_review_flags"] = scientific
    reviewed = {r["prompt_sha256"] for r in (review_exclusions or [])}
    for row in candidates:
        if digest(normalize(row["input"])) in reviewed:
            excluded.add(row["line"])
    detected["manual_review_exclusions"] = review_exclusions or []
    split = {name: [] for name in ("train", "valid", "test", "quarantine", "capped")}
    counts = defaultdict(int)
    for group in components(candidates):
        group_id = digest(sorted(r["id"] for r in group))
        if any(row["line"] in excluded for row in group):
            partition = "quarantine"
        else:
            bucket = int(digest([seed, group_id])[:16], 16) / 2**64
            partition = "test" if bucket < test_fraction else "valid" if bucket < test_fraction + valid_fraction else "train"
        counts[partition] += 1
        kept = defaultdict(int)
        for row in group:
            family = template(row["input"])
            target = partition
            if partition != "quarantine" and kept[family] >= cap:
                target = "capped"
            else:
                kept[family] += 1
            meta = {**row["metadata"], "group_id": group_id, "partition": target,
                    "declared_groups": row["declared_groups"],
                    "provenance": {"kind": "legacy_transformed", "origins": row["origins"],
                                   "license": "unknown", "review_status": "unreviewed"},
                    "eligibility": "candidate_only" if target in {"train", "valid", "test"} else "excluded"}
            messages = row.get("messages") or [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["input"]}, {"role": "assistant", "content": row["output"]}]
            split[target].append({"id": row["id"], "messages": messages, "metadata": meta})
    for name in split:
        split[name].sort(key=lambda r: r["id"])
    return split, {"input_records": len(rows), "unique_pairs": len(candidates),
                   "duplicate_pairs_collapsed": len(rows) - len(candidates),
                   "groups_by_partition": dict(counts), "leakage": detected}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, action="append", required=True)
    ap.add_argument("--eval-dir", type=Path, default=Path("evals"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid-fraction", type=float, default=.1)
    ap.add_argument("--test-fraction", type=float, default=.1)
    ap.add_argument("--max-per-template", type=int, default=3)
    ap.add_argument("--review-exclusions", type=Path, default=Path("data/manifests/firm3_review_exclusions.json"))
    args = ap.parse_args()
    if args.out_dir.exists():
        ap.error("Output directory exists; choose a new version to preserve artifacts")
    rows = []
    for path in sorted(args.input):
        for row in require_clean(path):
            row["origins"] = [{"path": str(path), "sha256": sha256(path), "line": row["line"]}]
            rows.append(row)
    exclusions = json.loads(args.review_exclusions.read_text())["exclusions"] if args.review_exclusions.exists() else []
    splits, summary = build_splits(rows, load_evals(args.eval_dir), args.seed,
                                  args.valid_fraction, args.test_fraction, args.max_per_template, exclusions)
    args.out_dir.mkdir(parents=True)
    for name, items in splits.items():
        write_jsonl(args.out_dir / (name + ".jsonl"), items)
    manifest = {"schema_version": "1.0", "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "tool_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("firm_data.py")]},
                "seed": args.seed, "method": "hash-assigned connected numeric prompt/response templates and declared families",
                "requested_fractions": {"valid": args.valid_fraction, "test": args.test_fraction},
                "max_per_template": args.max_per_template, "scientific_review": "pending; not a production release",
                "sources": [{"path": str(p), "sha256": sha256(p)} for p in sorted(args.input)],
                "eval_sources": [{"path": str(p), "sha256": sha256(p)} for p in sorted(args.eval_dir.glob("*.jsonl"))],
                "review_exclusions": {"path": str(args.review_exclusions), "sha256": sha256(args.review_exclusions)}
                                     if args.review_exclusions.exists() else None,
                "outputs": [{"path": name + ".jsonl", "sha256": sha256(args.out_dir / (name + ".jsonl")),
                             "records": len(items)} for name, items in splits.items()],
                "summary": summary}
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps({name: len(items) for name, items in splits.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
