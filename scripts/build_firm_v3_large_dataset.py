#!/usr/bin/env python3
"""Build a large, expert-strength FIRM v3 training set.

This script does NOT shrink the original dataset. It uses the original large CSV as
the base, applies selected rewrite overrides, appends all expert JSONL batches,
and optionally generates a deterministic synthetic expert expansion set.

Recommended use from repo root:

    python scripts/build_firm_v3_large_dataset.py \
      --base-csv "FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv" \
      --overrides data/curation/firm_v2_rewrite_overrides.jsonl \
      --expert-glob "data/processed/firm_v2_expert_hgcdte_deep_batch*.jsonl" \
      --synthetic-count 1000 \
      --out-csv data/processed/firm_v3_large_expert_dataset.csv \
      --out-jsonl data/processed/firm_v3_large_expert_sft.jsonl

Design goal
-----------
FIRM v3 should be large enough to retain the original broad formula/calc
coverage while adding deep analytical, empirical, and mathematical examples for
high-level infrared photonics engineers.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

COLUMNS = ["input", "output", "topic", "subtopic", "tags", "difficulty", "format"]

SYSTEM_PROMPT = (
    "You are FIRM, the Focused Infrared Research Model. Answer as a precise infrared "
    "photodetector research assistant specializing in HgCdTe/MCT, InSb, photoconductors, "
    "photodiodes, detector noise, blackbody radiometry, cryogenic testing, lock-in "
    "measurements, fabrication, passivation, and empirical troubleshooting. Use equations, "
    "units, assumptions, physical interpretation, measurement consequences, and caveats."
)

TOPIC_LIBRARY = [
    {
        "name": "HgCdTe bandgap and cutoff",
        "topic": "Physics",
        "subtopic": "HgCdTe Bandgap",
        "tags": "HgCdTe,MCT,bandgap,cutoff wavelength,composition,temperature,dark current",
        "questions": [
            "Explain why increasing HgCdTe cutoff wavelength usually increases dark current risk.",
            "Derive the connection between cutoff wavelength and thermal carrier population in HgCdTe.",
            "How should FIRM discuss composition extraction from HgCdTe cutoff wavelength?",
        ],
        "answer": (
            "Use E_g[eV] ≈ 1.2398/λ_c[µm] as the optical anchor. Longer cutoff wavelength implies "
            "smaller bandgap. Intrinsic carrier concentration scales roughly as n_i ∝ exp(-E_g/(2k_B T)), "
            "so small bandgap reductions can cause large increases in thermally generated carriers. In detector "
            "terms, those carriers increase dark conductance, shot noise, G-R noise, and reduce D*. Composition "
            "inference requires temperature, cutoff criterion, and an empirical Hg_{1-x}Cd_xTe bandgap model; "
            "FIRM should not give x as exact from λ_c alone."
        ),
    },
    {
        "name": "Photoconductive gain",
        "topic": "Devices",
        "subtopic": "Photoconductive Gain",
        "tags": "HgCdTe,photoconductor,gain,lifetime,transit time,mobility,bias,bandwidth",
        "questions": [
            "Explain the gain-bandwidth tradeoff in an HgCdTe photoconductor.",
            "Why can increasing bias increase photoconductor responsivity but not necessarily D*?",
            "Derive photoconductive gain from lifetime and transit time.",
        ],
        "answer": (
            "For a simple channel, transit time is τ_t = L^2/(µV_b), where L is contact spacing, µ is mobility, "
            "and V_b is bias. Photoconductive gain is approximately G = τ/τ_t = τµV_b/L^2. Increasing bias or lifetime "
            "can raise responsivity, but can also raise Joule heating, 1/f noise, contact injection, and field-dependent "
            "leakage. Bandwidth often follows f_3dB ≈ 1/(2πτ_eff), so long lifetime improves low-frequency gain while "
            "reducing speed. Expert evaluation compares R_V, noise density, D*, resistance, and frequency response together."
        ),
    },
    {
        "name": "Noise PSD decomposition",
        "topic": "Noise",
        "subtopic": "Noise Spectrum Analysis",
        "tags": "PSD,Johnson noise,shot noise,1/f noise,G-R noise,Lorentzian,HgCdTe",
        "questions": [
            "How should a FIRM-style answer decompose an HgCdTe noise spectrum?",
            "Given white noise, 1/f excess, and a Lorentzian hump, how should the spectrum be interpreted?",
            "What measurements distinguish detector noise from readout artifacts?",
        ],
        "answer": (
            "Model the calibrated spectrum as S(f)=S_white + A/f^α + Σ_i B_i/[1+(2πfτ_i)^2] plus narrow artifact lines. "
            "S_white may include Johnson, amplifier, and shot noise. A/f^α captures flicker behavior from traps, contacts, or "
            "mobility fluctuations. Lorentzians often indicate G-R or trapping processes with τ_i = 1/(2πf_c). Validate mechanisms "
            "by varying bias, temperature, illumination, grounding, and dummy loads. FIRM should avoid assigning physical origin from "
            "shape alone without controls."
        ),
    },
    {
        "name": "Blackbody calibration",
        "topic": "Radiometry",
        "subtopic": "Blackbody Calibration",
        "tags": "blackbody,Planck law,radiometry,HgCdTe,responsivity,optical power,etendue",
        "questions": [
            "Connect blackbody temperature to detector voltage in an HgCdTe test.",
            "What radiometric terms are needed to calculate optical power on an IR detector?",
            "Why is blackbody calibration dangerous if window transmission and solid angle are ignored?",
        ],
        "answer": (
            "The detector responds to coupled optical power, not source temperature directly. Use P = ∫ L_λ(T) ε(λ) τ_opt(λ) "
            "A_det Ω R_rel(λ) dλ over the relevant band, where L_λ is Planck spectral radiance, ε is source emissivity, τ_opt is "
            "window/filter/optics transmission, A_det is detector area, Ω is accepted solid angle, and R_rel is spectral weighting if used. "
            "Measured voltage is V_sig = R_V P_mod. Report source temperature, aperture, F-number or Ω, filter bandpass, detector area, "
            "modulation depth, and whether power is total, in-band, or response-weighted."
        ),
    },
    {
        "name": "Lock-in measurement discipline",
        "topic": "Testing",
        "subtopic": "Lock-in Measurement",
        "tags": "lock-in,MFLI,LabOne,chopper,bandwidth,time constant,HgCdTe,photoconductor",
        "questions": [
            "How should FIRM analyze a lock-in measurement of an HgCdTe photoconductor?",
            "Why is lock-in amplitude not automatically voltage responsivity?",
            "How do lock-in time constant and equivalent noise bandwidth affect detector metrics?",
        ],
        "answer": (
            "A lock-in reports demodulated X, Y, R, and phase relative to a reference. Responsivity requires calibrated modulated optical "
            "power at the detector: R_V = V_signal/P_mod. The noise depends on equivalent noise bandwidth set by time constant and filter order; "
            "RMS noise after filtering is not the same as V/√Hz. FIRM should track bias, detector resistance, preamp gain, input range, demodulator "
            "bandwidth, phase, chopper frequency, optical alignment, and dark/control signals before interpreting amplitude or D*."
        ),
    },
    {
        "name": "Surface and passivation",
        "topic": "Surface",
        "subtopic": "HgCdTe Passivation",
        "tags": "HgCdTe,passivation,surface states,1/f noise,leakage,perimeter current",
        "questions": [
            "Why is passivation central to HgCdTe detector noise and leakage?",
            "How can geometry scaling identify surface leakage in HgCdTe devices?",
            "A passivation process lowers noise but also lowers responsivity. How should this be interpreted?",
        ],
        "answer": (
            "HgCdTe surfaces can introduce trap states, surface recombination, band bending, lateral leakage, and conductance fluctuations. "
            "In photoconductors this often appears as 1/f noise, drift, and bias-history dependence. In photodiodes it may appear as perimeter leakage. "
            "Geometry scaling helps: bulk conductance follows roughly G=σWt/L, while surface leakage may scale with perimeter or exposed surface. If passivation "
            "reduces noise more than responsivity, D* can improve despite lower signal. Compare R_V, noise density, D*, optical transmission, and PSD before/after."
        ),
    },
    {
        "name": "Annealing and defects",
        "topic": "Fabrication",
        "subtopic": "HgCdTe Annealing",
        "tags": "HgCdTe,annealing,Hg overpressure,defects,carrier concentration,resistance,noise",
        "questions": [
            "Why can annealing change HgCdTe detector behavior without moving cutoff wavelength?",
            "How should FIRM interpret resistance and noise changes after HgCdTe annealing?",
            "What measurements separate optical cutoff stability from electrical improvement after anneal?",
        ],
        "answer": (
            "Cutoff wavelength mainly tracks bandgap/composition, but annealing can alter mercury vacancy concentration, dopant activation, compensation, "
            "point-defect distribution, surface stoichiometry, and contact behavior. Resistance, mobility, lifetime, G-R noise, and 1/f noise can change without "
            "large cutoff shift. Use Hall data, dark resistance versus temperature, noise PSD, spectral response, and contact IV before/after anneal. FIRM should not "
            "equate unchanged cutoff with unchanged device physics."
        ),
    },
    {
        "name": "Empirical inference guardrails",
        "topic": "Guardrails",
        "subtopic": "Empirical Inference",
        "tags": "HgCdTe,measurement,uncertainty,guardrail,empirical reasoning,model discipline",
        "questions": [
            "How should FIRM avoid overclaiming from one HgCdTe measurement?",
            "Why is one IV curve insufficient to prove material quality?",
            "What language should FIRM use when data are suggestive but not conclusive?",
        ],
        "answer": (
            "FIRM should distinguish evidence from conclusion. A single IV, spectrum, or noise trace rarely identifies a mechanism uniquely. Use language such as "
            "consistent with, suggests, rules against, or requires additional measurement. Material quality should be inferred from correlated optical spectra, Hall data, "
            "dark resistance versus temperature, responsivity, frequency response, noise PSD, geometry scaling, and controls. Avoid definitive defect, trap, or process claims "
            "unless the measurement set constrains alternatives."
        ),
    },
]

PARAM_SETS = [
    {"cutoff_um": 5.0, "temp_K": 77, "band": "MWIR"},
    {"cutoff_um": 8.0, "temp_K": 77, "band": "LWIR edge"},
    {"cutoff_um": 10.0, "temp_K": 77, "band": "LWIR"},
    {"cutoff_um": 12.0, "temp_K": 77, "band": "VLWIR"},
    {"cutoff_um": 5.0, "temp_K": 150, "band": "HOT MWIR"},
    {"cutoff_um": 10.0, "temp_K": 120, "band": "HOT LWIR"},
]


def read_legacy_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    header_row = None
    for idx, row in raw.iterrows():
        values = [str(v).strip().lower() for v in row.tolist()]
        if "input" in values and "output" in values:
            header_row = idx
            break
    if header_row is None:
        raise ValueError("Could not locate input/output header row")
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()
    return df[(df["input"] != "") & (df["output"] != "")][COLUMNS].drop_duplicates()


def read_jsonl_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
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
            if user and assistant:
                rows.append({
                    "input": user,
                    "output": assistant,
                    "topic": str(meta.get("topic", "Expert")),
                    "subtopic": str(meta.get("subtopic", "Expert Expansion")),
                    "tags": str(meta.get("tags", "FIRM,HgCdTe,infrared photodetectors")),
                    "difficulty": str(meta.get("difficulty", "Expert")),
                    "format": str(meta.get("format", "Analytical")),
                })
    return rows


def read_overrides(path: Path) -> Dict[str, Dict[str, str]]:
    overrides = {}
    if not path or not path.exists():
        return overrides
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            overrides[str(obj["match_input"]).strip()] = obj["replacement"]
    return overrides


def apply_overrides(df: pd.DataFrame, overrides: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    if not overrides:
        return df
    out = df.copy()
    for idx, row in out.iterrows():
        key = str(row["input"]).strip()
        if key in overrides:
            repl = overrides[key]
            for col in COLUMNS:
                out.at[idx, col] = repl.get(col, row[col])
    return out


def generate_synthetic_rows(count: int) -> List[Dict[str, str]]:
    rows = []
    i = 0
    while len(rows) < count:
        lib = TOPIC_LIBRARY[i % len(TOPIC_LIBRARY)]
        q = lib["questions"][(i // len(TOPIC_LIBRARY)) % len(lib["questions"])]
        p = PARAM_SETS[i % len(PARAM_SETS)]
        variant = i // (len(TOPIC_LIBRARY) * len(PARAM_SETS))
        prompt = f"{q} Use a {p['band']} HgCdTe context with cutoff near {p['cutoff_um']} µm at {p['temp_K']} K."
        answer = lib["answer"] + (
            f" In this specific context, λ_c ≈ {p['cutoff_um']} µm implies E_g ≈ {1.2398/p['cutoff_um']:.3f} eV. "
            f"At {p['temp_K']} K, any comparison must keep temperature fixed because thermal carrier density depends exponentially on E_g/(2k_B T). "
            f"Variant {variant}: emphasize that reported conclusions require device geometry, bias condition, optical calibration, and bandwidth."
        )
        rows.append({
            "input": prompt,
            "output": answer,
            "topic": lib["topic"],
            "subtopic": lib["subtopic"],
            "tags": lib["tags"] + f",{p['band']},{p['temp_K']}K,{p['cutoff_um']}um",
            "difficulty": "Expert",
            "format": "Synthetic Expert Expansion",
        })
        i += 1
    return rows


def to_sft(row: pd.Series) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(row["input"]).strip()},
            {"role": "assistant", "content": str(row["output"]).strip()},
        ],
        "metadata": {col: str(row[col]).strip() for col in ["topic", "subtopic", "tags", "difficulty", "format"]},
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
    ap.add_argument("--overrides", type=Path)
    ap.add_argument("--expert-glob", default="data/processed/firm_v2_expert_hgcdte_deep_batch*.jsonl")
    ap.add_argument("--synthetic-count", type=int, default=1000)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    args = ap.parse_args()

    base = read_legacy_csv(args.base_csv)
    base = apply_overrides(base, read_overrides(args.overrides) if args.overrides else {})

    expert_rows = []
    for p in sorted(glob.glob(args.expert_glob)):
        expert_rows.extend(read_jsonl_rows(Path(p)))

    synth_rows = generate_synthetic_rows(args.synthetic_count)
    merged = pd.concat([base, pd.DataFrame(expert_rows), pd.DataFrame(synth_rows)], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["input", "output"], keep="first").reset_index(drop=True)

    write_csv(merged, args.out_csv)
    write_jsonl(merged, args.out_jsonl)

    print(f"base_rows={len(base)}")
    print(f"expert_rows={len(expert_rows)}")
    print(f"synthetic_rows={len(synth_rows)}")
    print(f"dedup_removed={before-len(merged)}")
    print(f"final_rows={len(merged)}")
    print(f"out_csv={args.out_csv}")
    print(f"out_jsonl={args.out_jsonl}")


if __name__ == "__main__":
    main()
