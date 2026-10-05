"""Shared, dependency-free data inspection. Never silently drop bad records.

Numeric-template matching is a conservative structural heuristic, not a semantic
or scientific correctness classifier. Sources remain intact; callers decide policy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

FIELDS = ("input", "output", "topic", "subtopic", "tags", "difficulty", "format")
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize(text, case=True):
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    return text.casefold() if case else text


def template(text):
    return NUMBER.sub("<num>", normalize(text))


def words(text):
    return re.findall(r"\S+", text)


def tokens(text):
    return set(re.findall(r"\w+|<num>", normalize(text)))


def similarity(a, b):
    left, right = tokens(a), tokens(b)
    jaccard = len(left & right) / max(1, len(left | right))
    # Char similarity catches paraphrases and unit/punctuation variants.
    char = SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()
    return jaccard, char


def read_records(path, evaluation=False):
    """Return records and parse metadata; malformed entries remain in diagnostics.

    CSV location is the ending physical line of a logical record. JSONL location
    is its physical line. Rewrites are inspected as transformed candidate examples.
    """
    path = Path(path)
    records, malformed = [], []
    header, preamble, blanks = None, 0, 0
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                for row in reader:
                    if header is None:
                        lowered = [v.strip().lower() for v in row]
                        if "input" in lowered and "output" in lowered:
                            header = lowered
                        else:
                            preamble += 1
                        continue
                    if not row or not any(v.strip() for v in row):
                        blanks += 1
                        continue
                    if len(row) != len(header):
                        malformed.append({"line": reader.line_num, "reason": "CSV field count"})
                        continue
                    obj = dict(zip(header, row))
                    records.append({"input": obj.get("input", ""),
                                    "output": obj.get("output", ""),
                                    "metadata": {k: obj.get(k, "") for k in FIELDS[2:]},
                                    "line": reader.line_num, "raw": obj})
            except csv.Error as error:
                malformed.append({"line": reader.line_num, "reason": str(error)})
        if header is None:
            malformed.append({"line": 0, "reason": "No input/output header"})
    else:
        with path.open(encoding="utf-8-sig") as stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    blanks += 1
                    continue
                try:
                    obj = json.loads(text)
                    if not isinstance(obj, dict):
                        raise ValueError("Record must be an object")
                    row = obj.get("replacement", obj)
                    if not isinstance(row, dict):
                        raise ValueError("Replacement must be an object")
                    meta = row.get("metadata", {})
                    if not isinstance(meta, dict):
                        raise ValueError("metadata must be an object")
                    messages = row.get("messages", [])
                    if not isinstance(messages, list) or any(
                            not isinstance(m, dict) or m.get("role") not in
                            {"system", "user", "assistant", "tool"} or
                            not isinstance(m.get("content"), str) for m in messages):
                        raise ValueError("Unsupported messages (text-only audit)")
                    prompt = row.get("input", row.get("prompt", "\n".join(
                        m["content"] for m in messages if m["role"] == "user")))
                    answer = row.get("output", row.get("answer", "\n".join(
                        m["content"] for m in messages if m["role"] == "assistant")))
                    if not isinstance(prompt, str) or not isinstance(answer, str):
                        raise ValueError("Prompt and answer must be text")
                    records.append({"input": prompt, "output": answer,
                                    "metadata": {**{k: row[k] for k in FIELDS[2:] if k in row}, **meta},
                                    "messages": messages, "line": line, "raw": obj})
                except (ValueError, TypeError, KeyError) as error:
                    malformed.append({"line": line, "reason": str(error)})
    missing = [{"line": r["line"], "fields": [k for k in
                (("input",) if evaluation else ("input", "output")) if not r[k].strip()]}
               for r in records]
    missing = [m for m in missing if m["fields"]]
    return records, {"malformed": malformed, "missing_required": missing,
                     "csv_header": header, "preamble_rows": preamble, "blank_records": blanks}


def require_clean(path, evaluation=False):
    rows, info = read_records(path, evaluation)
    if info["malformed"] or info["missing_required"] or not rows:
        raise ValueError(f"Invalid source {path}: {info}")
    return rows


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    def percentile(p):
        idx = (len(values) - 1) * p
        low, high = math.floor(idx), math.ceil(idx)
        return values[low] + (values[high] - values[low]) * (idx - low)
    return {"count": len(values), "min": values[0], "mean": sum(values) / len(values),
            "p05": percentile(.05), "median": percentile(.5), "p95": percentile(.95),
            "p99": percentile(.99), "max": values[-1]}


def duplicate_summary(values):
    counts = Counter(values)
    return {"extra_records": sum(c - 1 for c in counts.values()),
            "groups": sum(c > 1 for c in counts.values()),
            "participating_records": sum(c for c in counts.values() if c > 1)}


def near_pairs(rows, field, threshold=.85, limit=20):
    """Exhaustive token-set Jaccard >= threshold via an inverted index.

    Counts include multiplicities of distinct normalized strings. Exact normalized
    duplicates are reported separately. Token order is ignored; no embeddings.
    """
    unique = defaultdict(list)
    for r in rows:
        unique[normalize(r[field])].append(r["line"])
    texts = sorted(unique)
    sets = [tokens(t) for t in texts]
    index, examples, pair_count = defaultdict(list), [], 0
    for i, left in enumerate(sets):
        overlaps = Counter(j for token in left for j in index[token])
        for j, overlap in overlaps.items():
            score = overlap / max(1, len(left) + len(sets[j]) - overlap)
            if score >= threshold:
                pair_count += len(unique[texts[i]]) * len(unique[texts[j]])
                if len(examples) < limit:
                    examples.append({"lines": [unique[texts[j]][0], unique[texts[i]][0]],
                                     "jaccard": score})
        for token in left:
            index[token].append(i)
    return {"threshold": threshold, "pair_count": pair_count, "examples": examples,
            "method": "exhaustive token-set Jaccard over case/whitespace-normalized texts"}


FEATURES = {
    "equation_like": r"[=∝≈]|\b(?:exp|sqrt)\s*\(|\\(?:frac|sqrt)",
    "numerical_problem": r"\d.*(?:compute|calculate|estimate|derive)|(?:compute|calculate|estimate).*\d",
    "derivation": r"\bderiv(?:e|ation|ing)\b",
    "troubleshooting": r"troubleshoot|diagnos|artifact|artefact|de.embed|plausible cause|control measurement",
    "empirical_data_mention": r"measured|measurement|\bdata\b|\bPSD\b|I.?V curve|Hall",
    "citation_like": r"https?://|doi\s*:|\bet al\b|\[\d+\]",
    "units_with_number": r"\d\s*(?:eV|K\b|Hz|[munµμ]?s\b|[munµμ]?m\b|Ohm|Ω|[munµμ]?[AVW]\b|cm\^)",
    "uncertainty": r"uncertain|error bar|confidence interval|propagat|Monte Carlo|caveat",
    "HgCdTe_MCT": r"HgCdTe|\bMCT\b|mercury cadmium telluride",
    "InSb": r"\bInSb\b", "InGaAs": r"\bInGaAs\b", "InAsSb": r"\bInAsSb\b",
    "T2SL": r"\bT2SL\b|type.?II superlattice", "MBE": r"\bMBE\b|molecular beam epitaxy",
    "MOCVD": r"\bMOCVD\b", "LPE": r"\bLPE\b|liquid.phase epitaxy",
}


def audit_records(rows, tokenizer=None):
    prompts = [r["input"] for r in rows]
    answers = [r["output"] for r in rows]
    joined = [r["input"] + "\n" + r["output"] + "\n" + str(r["metadata"]) for r in rows]
    duplicates = {}
    for mode, fn in [("exact", lambda t: t), ("whitespace", lambda t: normalize(t, False)),
                     ("case_unicode", normalize), ("numeric_template", template)]:
        p, a = list(map(fn, prompts)), list(map(fn, answers))
        duplicates[mode] = {"prompts": duplicate_summary(p), "responses": duplicate_summary(a),
                            "pairs": duplicate_summary(list(zip(p, a)))}
    families = defaultdict(list)
    contradictions = defaultdict(set)
    for row in rows:
        families[template(row["input"])].append(row["line"])
        contradictions[normalize(row["input"])].add(normalize(row["output"]))
    metrics = {"duplicates": duplicates,
               "missing_metadata_cells": {k: sum(not str(r["metadata"].get(k, "")).strip() for r in rows) for k in FIELDS[2:]},
               "prompt_words": distribution([len(words(p)) for p in prompts]),
               "response_words": distribution([len(words(a)) for a in answers]),
               "short_response_lt_6_words": [r["line"] for r in rows if len(words(r["output"])) < 6],
               "long_response_gt_300_words": [r["line"] for r in rows if len(words(r["output"])) > 300],
               "near_prompts": near_pairs(rows, "input"), "near_responses": near_pairs(rows, "output"),
               "top_prompt_templates": [{"template": t, "count": len(lines), "example_line": lines[0]}
                                         for t, lines in sorted(families.items(), key=lambda x: (-len(x[1]), x[0]))[:20]],
               "same_prompt_different_responses": [{"prompt": p, "answer_variants": len(a)}
                    for p, a in contradictions.items() if len(a) > 1],
               "feature_heuristics": {name: sum(bool(re.search(pat, j, re.I | re.S)) for j in joined)
                                      for name, pat in FEATURES.items()},
               "multi_turn": sum(sum(m["role"] == "user" for m in r.get("messages", [])) > 1 for r in rows),
               "scientific_calculation_checks": calculation_checks(rows),
               "category_balance": {key: dict(sorted(Counter(str(r["metadata"].get(key, "<missing>"))
                                                            for r in rows).items()))
                                    for key in ("topic", "subtopic", "difficulty", "format")}}
    if tokenizer is not None:
        metrics["prompt_tokens"] = distribution([len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts])
        metrics["response_tokens"] = distribution([len(tokenizer.encode(a, add_special_tokens=False)) for a in answers])
        metrics["chat_tokens"] = distribution([len(tokenizer.apply_chat_template(
            r.get("messages") or [{"role": "user", "content": r["input"]},
                                   {"role": "assistant", "content": r["output"]}],
            tokenize=True, add_generation_prompt=False)) for r in rows])
    return metrics


def calculation_checks(rows):
    """Check seven recognized legacy numerical families, without general prose grading.

    Tolerance 2% allows printed input/answer rounding. Failure means a review flag,
    not a universal assessment of physical validity. Critical distinction: D* in
    Jones requires cm area units, while these prompts supply m^2.
    """
    num = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    q, kb, h, c = 1.602176634e-19, 1.380649e-23, 6.62607015e-34, 299792458
    checks = []
    for row in rows:
        prompt, answer = normalize(row["input"], False), normalize(row["output"], False)
        expected, family, unit = None, None, None
        match = re.fullmatch(r"Given noise current " + num + r" A and responsivity " + num + r" A/W, compute the noise equivalent power \(NEP\)\.", prompt)
        if match:
            current, resp = map(float, match.groups())
            expected, family, unit = current/resp, "NEP_integrated", "W"
        match = re.fullmatch(r"A detector with area " + num + r" m\^2 and bandwidth " + num + r" Hz has NEP " + num + r" W\. Compute the specific detectivity \(D\*\)\.", prompt)
        if match:
            area, bandwidth, nep = map(float, match.groups())
            expected, family, unit = 100*math.sqrt(area*bandwidth)/nep, "Dstar_Jones", "Jones"
        match = re.fullmatch(r"Calculate the shot noise current for a photodiode with average current " + num + r" A in a bandwidth " + num + r" Hz\.", prompt)
        if match:
            current, bandwidth = map(float, match.groups())
            expected, family, unit = math.sqrt(2*q*current*bandwidth), "shot_noise_RMS", "A"
        match = re.fullmatch(r"Calculate the Johnson noise voltage for a resistor of " + num + r" (?:Ohm|Ω) at " + num + r" K in a bandwidth " + num + r" Hz\.", prompt)
        if match:
            resistance, temperature, bandwidth = map(float, match.groups())
            expected, family, unit = math.sqrt(4*kb*temperature*resistance*bandwidth), "Johnson_RMS", "V"
        match = re.fullmatch(r"Compute the shot-noise-limited SNR for a photodiode with responsivity " + num + r" A/W, optical power " + num + r" W, and bandwidth " + num + r" Hz\.", prompt)
        if match:
            resp, power, bandwidth = map(float, match.groups())
            current = resp*power
            expected, family, unit = current/math.sqrt(2*q*current*bandwidth), "shot_limited_SNR", "1"
        match = re.fullmatch(r"Compute the spectral radiance at " + num + r" K and wavelength " + num + r" nm using Planck[’']s law\.", prompt)
        if match:
            temperature, wavelength = map(float, match.groups())
            wavelength *= 1e-9
            expected = 2*h*c*c/(wavelength**5*math.expm1(h*c/(wavelength*kb*temperature)))
            family, unit = "Planck_radiance_per_m", "W/(m^2*sr*m)"
        match = re.fullmatch(r"A photodiode with quantum efficiency " + num + r" operates at " + num + r" nm\. If it receives " + num + r" mW of optical power, what is the photocurrent\?", prompt)
        if match:
            efficiency, wavelength, power = map(float, match.groups())
            expected, family, unit = efficiency*q*wavelength*1e-9/(h*c)*power*1e-3, "photocurrent", "A"
        if expected is None:
            continue
        values = (re.findall(r"≈\s*" + num + r"\s*W", answer)
                  if family == "Planck_radiance_per_m" else NUMBER.findall(answer))
        if not values:
            checks.append({"line": row["line"], "family": family, "status": "unparsed"})
            continue
        # These recognized legacy answers put the final reported value last.
        reported = float(values[-1])
        relative_error = abs(reported-expected)/abs(expected)
        checks.append({"line": row["line"], "family": family, "expected": expected, "unit": unit,
                       "reported": reported, "relative_error": relative_error,
                       "status": "pass" if relative_error <= .02 else "review"})
    return {"method": "seven exact legacy prompt families; final-number extraction; SI constants; 2% rounding tolerance",
            "count": len(checks), "by_family": dict(Counter(c["family"] for c in checks)),
            "status": dict(Counter(c["status"] for c in checks)),
            "flagged": [c for c in checks if c["status"] != "pass"]}


def leakage(rows, eval_items, fuzzy_threshold=.8, jaccard_threshold=.65):
    """High-recall lexical/structural candidates requiring scientific review.

    Whole-answer matches require >= 8 words; rubric phrases alone are not answers.
    No result proves absence of semantic leakage or foundation-model exposure.
    """
    findings, closest = [], []
    for item in eval_items:
        candidates = []
        for row in rows:
            a, b = row["input"], item["input"]
            reasons = []
            if a == b:
                reasons.append("exact_prompt")
            elif normalize(a) == normalize(b):
                reasons.append("normalized_prompt")
            elif template(a) == template(b):
                reasons.append("numeric_template")
            ja = len(tokens(a) & tokens(b)) / max(1, len(tokens(a) | tokens(b)))
            char = (SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()
                    if ja >= .15 else 0.0)
            tj = len(tokens(template(a)) & tokens(template(b))) / max(1, len(tokens(template(a)) | tokens(template(b))))
            if not reasons and (ja >= jaccard_threshold or char >= fuzzy_threshold or tj >= jaccard_threshold):
                reasons.append("fuzzy_or_structural_review")
            if len(words(item["output"])) >= 8 and normalize(row["output"]) == normalize(item["output"]):
                reasons.append("normalized_answer")
            rec = {"source_line": row["line"], "eval_id": item["raw"].get("id"),
                   "eval_source": item.get("source"), "jaccard": ja,
                   "char_similarity": char, "template_jaccard": tj, "reasons": reasons}
            candidates.append(rec)
            if reasons:
                findings.append(rec)
        closest.extend(sorted(candidates, key=lambda x: max(x["jaccard"], x["char_similarity"],
                                                         x["template_jaccard"]), reverse=True)[:3])
    return {"findings": findings, "closest_per_eval": closest,
            "thresholds": {"char": fuzzy_threshold, "token_jaccard": jaccard_threshold},
            "limitations": "Lexical and numeric-template candidates only; manual semantic review still required."}


def load_evals(root):
    items = []
    for path in sorted(Path(root).glob("*.jsonl")):
        for row in require_clean(path, evaluation=True):
            row["source"] = str(path)
            items.append(row)
    return items


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
