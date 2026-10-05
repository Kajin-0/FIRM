"""CPU regression checks for leakage, grouping, units and oracle math. No model scores."""
import json
import io
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from firm_data import calculation_checks, digest, leakage, normalize, read_records, require_clean, template
from split_firm_data import build_splits
from eval_firm_science import grade_quantity, score, validate_benchmark
from build_firm_numeric_pilot import bisect_composition, hsc_gap, pilot_items
from run_firm_baseline import messages_for, parse_answer
import run_firm_baseline
from train_firm_qlora import preflight


def row(prompt, answer, line=1, metadata=None):
    return {"input": prompt, "output": answer, "line": line, "metadata": metadata or {}, "raw": {}}


class FoundationTests(unittest.TestCase):
    def test_csv_preamble_multiline_and_missing_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "sample.csv"
            p.write_text(',\ninput,output\n"a\nb",answer\nmissing,\nwrong,width,extra\n')
            rows, info = read_records(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["input"], "a\nb")
            self.assertEqual(info["preamble_rows"], 1)
            self.assertEqual(len(info["missing_required"]), 1)
            self.assertEqual(len(info["malformed"]), 1)
            with self.assertRaises(ValueError):
                require_clean(p)

    def test_jsonl_bad_types_and_multiturn_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "sample.jsonl"
            p.write_text('[]\n{"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},'
                         '{"role":"user","content":"three"},{"role":"assistant","content":"four"}]}\n')
            rows, info = read_records(p)
            self.assertEqual(len(info["malformed"]), 1)
            self.assertEqual(rows[0]["input"], "one\nthree")
            self.assertEqual(len(rows[0]["messages"]), 4)

    def test_unicode_numeric_variants(self):
        self.assertEqual(normalize("  0.05\u00a0eV "), "0.05 ev")
        self.assertEqual(template("Calculate at 1.2e-3 W."), template("calculate at 8.5e-7 W."))

    def test_exact_normalized_template_and_fuzzy_leakage(self):
        example = row("Calculate noise at 10 K", "", 1)
        example["raw"] = {"id": "heldout"}
        rows = [row("Calculate noise at 10 K", "different", 1),
                row("  CALCULATE noise at 10 K", "different", 2),
                row("Calculate noise at 20 K", "different", 3)]
        reasons = [f["reasons"][0] for f in leakage(rows, [example])["findings"]]
        self.assertEqual(reasons, ["exact_prompt", "normalized_prompt", "numeric_template"])

    def test_split_deterministic_disjoint_and_family_quarantine(self):
        rows = [row(f"Compute value at {i} K", "full equation and detailed scientific answer with units here", i) for i in range(1, 5)]
        rows += [row(f"Distinct topic word_{chr(97+i)}", f"Independent answer number_{chr(97+i)}", i+10) for i in range(50)]
        heldout = row("Compute value at 9 K", "")
        heldout["raw"] = {"id": "eval"}
        first, summary = build_splits(rows, [heldout])
        second, _ = build_splits(list(reversed(rows)), [heldout])
        self.assertEqual(first, second)
        self.assertEqual(len(first["quarantine"]), 4)
        groups = [{r["metadata"]["group_id"] for r in first[p]} for p in ("train", "valid", "test")]
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i)))

    def test_manual_exclusion(self):
        rows = [row("Interpret this unexplained mechanism", "full explanation", 1)]
        splits, _ = build_splits(rows, [], review_exclusions=[{"prompt_sha256": digest(normalize(rows[0]["input"]))}])
        self.assertEqual(len(splits["quarantine"]), 1)

    def test_split_preserves_unit_case_and_duplicate_document_links(self):
        rows = [row("Given power 1 mW", "value 1 mW"), row("Given power 1 MW", "value 1 MW")]
        splits, summary = build_splits(rows, [])
        self.assertEqual(summary["unique_pairs"], 2)
        rows = [row("Same prompt", "Same answer", metadata={"document_id": "paper-a"}),
                row("Same prompt", "Same answer", metadata={"document_id": "paper-b"}),
                row("Other prompt", "Other answer", metadata={"document_id": "paper-b"})]
        first, summary = build_splits(rows, [])
        second, _ = build_splits(list(reversed(rows)), [])
        self.assertEqual(first, second)
        self.assertEqual(summary["unique_pairs"], 2)
        groups = {r["metadata"]["group_id"] for part in first.values() for r in part}
        self.assertEqual(len(groups), 1)

    def test_dstar_factor_100_regression(self):
        source = row("A detector with area 1.000e-08 m^2 and bandwidth 10000 Hz has NEP 1.000e-10 W. Compute the specific detectivity (D*).",
                     "D* = sqrt(A*df)/NEP = 1.000e+08 Jones.")
        check = calculation_checks([source])["flagged"][0]
        self.assertAlmostEqual(check["expected"], 1e10)
        self.assertAlmostEqual(check["relative_error"], .99)
        split, _ = build_splits([source], [])
        self.assertEqual(len(split["quarantine"]), 1)

    def test_radiance_exponent_not_reported_number(self):
        source = row("Compute the spectral radiance at 793.3 K and wavelength 1902.1 nm using Planck’s law.",
                     "Planck’s law gives L_λ ≈ 3.440e+08 W·sr⁻¹·m⁻³.")
        self.assertEqual(calculation_checks([source])["status"], {"pass": 1})

    def test_units_density_rms_and_detectivity(self):
        gold = {"value": 2e-6, "unit": "s", "rtol": .001, "atol": 0}
        self.assertTrue(grade_quantity(gold, {"value": 2, "unit": "µs"})["numerical_correct"])
        self.assertFalse(grade_quantity(gold, {"value": 2e-6, "unit": "Hz"})["unit_correct"])
        gold = {"value": 1e10, "unit": "Jones", "rtol": .001, "atol": 0}
        self.assertTrue(grade_quantity(gold, {"value": 1e8, "unit": "m*sqrt(Hz)/W"})["numerical_correct"])
        gold = {"value": 1e-9, "unit": "V/sqrt(Hz)", "rtol": .001, "atol": 0}
        self.assertFalse(grade_quantity(gold, {"value": 1e-9, "unit": "V"})["unit_correct"])

    def test_invalid_predictions_fail_or_score_zero(self):
        gold = {"value": 1, "unit": "1", "rtol": .01, "atol": 0}
        for invalid in [None, {"value": float("nan"), "unit": "1"}, {"value": True, "unit": "1"},
                        {"value": 1, "unit": []}]:
            self.assertFalse(grade_quantity(gold, invalid)["numerical_correct"])
        item = pilot_items()[0]
        self.assertEqual(score([item], [])["numerical_accuracy"], 0)
        with self.assertRaises(ValueError):
            score([item], [{"id": item["id"]}, {"id": item["id"]}])
        with self.assertRaises(ValueError):
            score([item], [{"id": "unknown"}])

    def test_pilot_oracles_analytic_limits(self):
        items = pilot_items()
        validate_benchmark(items)
        by_id = {i["id"]: i for i in items}
        slab = by_id["numeric_pilot_007"]["grading"]["quantities"]
        self.assertAlmostEqual(slab["thickness_95pct"]["value"], 29.9573227355)
        self.assertAlmostEqual(slab["absorption_20um"]["value"], .86466471676)
        enbw = by_id["numeric_pilot_008"]["grading"]["quantities"]
        self.assertAlmostEqual(enbw["enbw"]["value"], 12.5)
        self.assertAlmostEqual(enbw["rms_noise"]["value"], 4.242640687119285e-8)
        self.assertAlmostEqual(hsc_gap(bisect_composition(.13, 85), 85), .13, places=12)

    def test_pilot_oracles_independent_numeric_integration(self):
        # Midpoint integration independently checks both closed-form PSD integrals.
        n = 20000
        lo, hi, coefficient, white = 2, 500, 4e-16, 9e-18
        width = (hi-lo)/n
        integral = sum(coefficient/(lo+(i+.5)*width) + white for i in range(n))*width
        expected = coefficient*math.log(hi/lo) + white*(hi-lo)
        self.assertLess(abs(integral-expected)/expected, 1e-5)
        # f=tan(theta)/(2*pi*tau) maps infinite range to 0..pi/2.
        tau, dtheta = .02, math.pi/(2*n)
        integral = sum((1/(1+math.tan((i+.5)*dtheta)**2)) *
                       (1/math.cos((i+.5)*dtheta)**2)/(2*math.pi*tau) for i in range(n))*dtheta
        self.assertAlmostEqual(integral, 1/(4*tau), places=10)

    def test_no_gold_in_generation_protocol(self):
        item = pilot_items()[0]
        serialized = json.dumps(messages_for(item))
        self.assertNotIn(str(item["grading"]["quantities"]["composition_x"]["value"]), serialized)
        self.assertNotIn("manual_rubric", serialized)
        self.assertEqual(parse_answer("bad JSON")["quantities"], {})

    def test_existing_eval_hashes_and_pilot_schema(self):
        from firm_data import sha256
        for path in ROOT.glob("evals/*_manifest.json"):
            manifest = json.loads(path.read_text())
            for asset in manifest.get("assets", []):
                self.assertEqual(sha256(ROOT / asset["path"]), asset["sha256"])
        schema = json.loads((ROOT / "evals/firm_benchmark_schema.json").read_text())
        try:
            import jsonschema
        except ImportError:
            return  # Runtime validator above remains dependency-free.
        for item in pilot_items():
            jsonschema.validate(item, schema)

    def test_training_preflight_requires_hashes_without_gpu_import(self):
        from firm_data import sha256, write_json, write_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            write_jsonl(train, [{"messages": [{"role": "user", "content": "fixture prompt"},
                                              {"role": "assistant", "content": "fixture answer"}],
                                 "metadata": {"partition": "train"}}])
            manifest = root / "manifest.json"
            write_json(manifest, {"outputs": [{"path": "train.jsonl", "sha256": sha256(train)}]})
            evaluation = root / "eval.jsonl"
            write_jsonl(evaluation, [pilot_items()[0]])
            frozen = root / "eval_manifest.json"
            write_json(frozen, {"assets": [{"path": str(evaluation), "sha256": sha256(evaluation)}]})
            args = SimpleNamespace(model="Qwen/Qwen3-0.6B", model_revision="0"*40,
                dataset_manifest=manifest, eval_manifest=[frozen], train=str(train), valid=None,
                out=str(root / "out"), run_id="test-fixture", seed=42, config=None,
                max_seq_length=512, batch_size=1, grad_accum=4, lora_r=8, lora_alpha=16,
                lora_dropout=.05, lr=2e-4, epochs=1, max_steps=10, save_steps=2, eval_steps=2,
                max_train_examples=1, resume_from_checkpoint=None)
            run = preflight(args)
            self.assertEqual(run["training_loss"], None)
            try:
                import jsonschema
            except ImportError:
                pass
            else:
                schema = json.loads((ROOT / "data/manifests/experiment_manifest_schema.json").read_text())
                jsonschema.validate({**run, "status": "dry_run_no_training"}, schema)
            train.write_text(train.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "Dataset hash mismatch"):
                preflight(args)

    def test_baseline_interrupted_run_resumes_without_duplicate_predictions(self):
        from firm_data import write_jsonl
        import urllib.error
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, out = root / "eval.jsonl", root / "fixture.jsonl"
            write_jsonl(evaluation, pilot_items()[:2])
            argv = ["run_firm_baseline.py", "--eval", str(evaluation), "--model", "fixture/no-weights",
                    "--model-revision", "0"*40, "--out", str(out)]
            def response():
                return io.BytesIO(json.dumps({"choices": [{"message": {"content":
                    json.dumps({"answer": "test fixture only", "quantities": {}})}, "finish_reason": "stop"}]}).encode())
            with patch.object(sys, "argv", argv), patch.object(run_firm_baseline.subprocess, "check_output", return_value="0"*40), \
                    patch.object(run_firm_baseline.urllib.request, "urlopen", side_effect=[response(), urllib.error.URLError("fixture interruption")]):
                with self.assertRaises(urllib.error.URLError):
                    run_firm_baseline.main()
            self.assertEqual(len(out.read_text().splitlines()), 1)
            with patch.object(sys, "argv", argv), patch.object(run_firm_baseline.subprocess, "check_output", return_value="0"*40), \
                    patch.object(run_firm_baseline.urllib.request, "urlopen", return_value=response()) as request:
                run_firm_baseline.main()
                self.assertEqual(request.call_count, 1)
            results = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len({r["id"] for r in results}), 2)
            self.assertEqual(json.loads(out.with_suffix(".run.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
