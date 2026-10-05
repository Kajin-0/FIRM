#!/usr/bin/env python3
"""Deterministic structured numerical grading; physical reasoning remains manual.

Prediction: {"id": "...", "answer": "...", "quantities":
             {"tau": {"value": 150, "unit": "us"}}}
Only declared quantities are graded. No number mining from prose or LLM judge.
"""
import argparse
import json
import math
from pathlib import Path

from firm_data import require_clean, sha256, write_json

# dimension, factor to a common base; explicitly limited audited unit vocabulary.
UNITS = {
    "1": ("dimensionless", 1), "": ("dimensionless", 1),
    "dimensionless": ("dimensionless", 1),
    "s": ("time", 1), "ms": ("time", 1e-3), "us": ("time", 1e-6), "ns": ("time", 1e-9),
    "Hz": ("frequency", 1), "kHz": ("frequency", 1e3),
    "m": ("length", 1), "cm": ("length", .01), "um": ("length", 1e-6),
    "m^2": ("area", 1), "cm^2": ("area", 1e-4),
    "K": ("temperature", 1), "eV": ("energy", 1.602176634e-19), "J": ("energy", 1),
    "Ohm": ("resistance", 1), "kOhm": ("resistance", 1e3),
    "Ohm*m": ("resistivity", 1), "Ohm*cm": ("resistivity", .01),
    "V": ("voltage", 1), "mV": ("voltage", 1e-3), "uV": ("voltage", 1e-6), "nV": ("voltage", 1e-9),
    "A": ("current", 1), "mA": ("current", 1e-3), "uA": ("current", 1e-6), "nA": ("current", 1e-9),
    "W": ("power", 1), "mW": ("power", 1e-3), "uW": ("power", 1e-6), "nW": ("power", 1e-9),
    "A/W": ("current_responsivity", 1), "V/W": ("voltage_responsivity", 1),
    "V/sqrt(Hz)": ("voltage_ASD", 1), "nV/sqrt(Hz)": ("voltage_ASD", 1e-9),
    "A/sqrt(Hz)": ("current_ASD", 1), "pA/sqrt(Hz)": ("current_ASD", 1e-12),
    "W/sqrt(Hz)": ("NEP_ASD", 1), "pW/sqrt(Hz)": ("NEP_ASD", 1e-12),
    "V^2/Hz": ("voltage_PSD", 1), "A^2/Hz": ("current_PSD", 1),
    "Jones": ("detectivity", .01), "cm*sqrt(Hz)/W": ("detectivity", .01),
    "m*sqrt(Hz)/W": ("detectivity", 1),
    "W/(m^2*sr*m)": ("spectral_radiance_lambda", 1),
    "W/(m^2*sr*um)": ("spectral_radiance_lambda", 1e6),
    "S/m": ("conductivity", 1), "S/cm": ("conductivity", 100),
    "m^3/C": ("Hall_coefficient", 1), "cm^3/C": ("Hall_coefficient", 1e-6),
    "m^-3": ("number_density", 1), "cm^-3": ("number_density", 1e6),
    "m^2/(V*s)": ("mobility", 1), "cm^2/(V*s)": ("mobility", 1e-4),
    "deg": ("angle", math.pi/180), "rad": ("angle", 1),
    "sr": ("solid_angle", 1), "um/K": ("cutoff_temperature_slope", 1e-6),
    "m/K": ("cutoff_temperature_slope", 1),
    "A^2": ("current_variance", 1), "V^2": ("voltage_variance", 1),
    "W/(m^2*sr)": ("band_radiance", 1),
    "s^-1": ("rate", 1), "1/s": ("rate", 1),
    "cm^-1": ("inverse_length", 100), "m^-1": ("inverse_length", 1),
}


def unit_key(unit):
    if not isinstance(unit, str):
        raise ValueError("Unit must be text")
    return unit.replace("µ", "u").replace("μ", "u").replace("Ω", "Ohm").replace(" ", "")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def grade_quantity(expected, predicted):
    result = {"present": predicted is not None, "unit_correct": False,
              "numerical_correct": False, "relative_error": None}
    if not isinstance(predicted, dict) or not finite_number(predicted.get("value")):
        return {**result, "error": "missing or non-finite quantity"}
    expected_unit = unit_key(expected["unit"])
    try:
        predicted_unit = unit_key(predicted.get("unit", ""))
    except ValueError:
        return {**result, "error": "invalid unit type"}
    if expected_unit not in UNITS:
        raise ValueError("Unsupported gold unit: " + expected_unit)
    if predicted_unit not in UNITS or UNITS[predicted_unit][0] != UNITS[expected_unit][0]:
        return {**result, "error": "unknown unit or wrong dimension"}
    value = predicted["value"] * UNITS[predicted_unit][1] / UNITS[expected_unit][1]
    error = abs(value - expected["value"])
    result.update({"unit_correct": True, "converted_value": value,
                   "absolute_error": error,
                   "relative_error": error / abs(expected["value"]) if expected["value"] else None,
                   "numerical_correct": error <= expected.get("atol", 0) + expected.get("rtol", .01) * abs(expected["value"])})
    return result


def validate_benchmark(items):
    ids = set()
    for item in items:
        if item.get("schema_version") != "1.0" or not item.get("id") or item["id"] in ids:
            raise ValueError("Missing schema version or duplicate/empty benchmark ID")
        ids.add(item["id"])
        for key in ("category", "prompt", "provenance", "grading"):
            if key not in item:
                raise ValueError("Missing benchmark field " + key)
        grading = item["grading"]
        if not isinstance(grading.get("quantities"), dict) or not isinstance(grading.get("manual_rubric"), list):
            raise ValueError("Invalid grading object")
        for quantity in grading["quantities"].values():
            if not finite_number(quantity.get("value")) or unit_key(quantity.get("unit", "")) not in UNITS:
                raise ValueError("Invalid gold quantity")
            if any(not finite_number(quantity.get(k)) or quantity[k] < 0 for k in ("rtol", "atol")):
                raise ValueError("Invalid numerical tolerance")


def score(items, predictions):
    validate_benchmark(items)
    by_id = {}
    for row in predictions:
        if not isinstance(row, dict) or "id" not in row or row["id"] in by_id:
            raise ValueError("Missing or duplicate prediction ID")
        if not isinstance(row.get("answer", ""), str) or not isinstance(row.get("quantities", {}), dict):
            raise ValueError("Invalid prediction answer/quantities")
        by_id[row["id"]] = row
    if set(by_id) - {r["id"] for r in items}:
        raise ValueError("Unexpected prediction IDs")
    results = []
    for item in items:
        prediction = by_id.get(item["id"], {})
        quantities = {name: grade_quantity(gold, prediction.get("quantities", {}).get(name))
                      for name, gold in item["grading"]["quantities"].items()}
        results.append({"id": item["id"], "category": item["category"],
                        "prediction_present": item["id"] in by_id, "quantities": quantities,
                        "manual_metrics": {name: None for name in ("physical_correctness", "equation_correctness",
                            "derivation_validity", "reasoning_completeness", "diagnosis_quality",
                            "citation_correctness", "unsupported_claim_rate")},
                        "manual_rubric": item["grading"]["manual_rubric"]})
    all_quantities = [q for r in results for q in r["quantities"].values()]
    n = len(all_quantities)
    return {"schema_version": "1.0", "items": results, "quantity_count": n,
            "prediction_coverage": len(by_id) / len(items) if items else None,
            "numerical_accuracy": sum(q["numerical_correct"] for q in all_quantities) / n if n else None,
            "unit_correctness": sum(q["unit_correct"] for q in all_quantities) / n if n else None,
            "manual_status": "not scored; requires independent scientific review",
            "limitations": "Structured quantities only; correct numbers do not establish correct physics or derivation."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="Optional frozen benchmark manifest; auto-discovered beside the eval file")
    args = ap.parse_args()
    items = [r["raw"] for r in require_clean(args.eval, evaluation=True)]
    predictions = [json.loads(line) for line in args.pred.read_text().splitlines() if line.strip()]
    eval_hash = sha256(args.eval)
    manifest_path = args.manifest or args.eval.with_name(args.eval.stem + "_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        matches = [a for a in manifest["assets"] if Path(a["path"]).resolve() == args.eval.resolve()]
        if len(matches) != 1 or matches[0]["sha256"] != eval_hash:
            raise ValueError("Frozen benchmark hash mismatch")
    for prediction in predictions:
        if prediction.get("eval_sha256", eval_hash) != eval_hash:
            raise ValueError("Prediction was generated for a different eval hash")
    report = score(items, predictions)
    report["eval_sha256"] = sha256(args.eval)
    report["predictions_sha256"] = sha256(args.pred)
    report["eval_manifest_sha256"] = sha256(manifest_path) if manifest_path.exists() else None
    run_path = args.pred.with_suffix(".run.json")
    if run_path.exists():
        run = json.loads(run_path.read_text())
        if run["spec"]["eval_sha256"] != eval_hash:
            raise ValueError("Prediction run manifest eval mismatch")
        report["generation_manifest"] = {"path": str(run_path), "sha256": sha256(run_path), "status": run["status"], "spec": run["spec"]}
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"items"}}, indent=2))


if __name__ == "__main__":
    main()
