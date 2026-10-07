"""Integrity gates for FIRM-4B v2 curriculum and trainer."""
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm4b_v2_foundation import build,FACTS
from train_firm4b_v2 import get_args,verify_data,run,DATA_DEFAULT
from firm_model_profiles import QWEN35_4B_REVISION

class Firm4BV2Tests(unittest.TestCase):
    def test_checked_foundational_cards_cover_v1_failures(self):
        answers={fid:a for fid,q,a,src in FACTS}
        self.assertIn("mercury cadmium telluride",answers["mct_identity"].lower())
        self.assertIn("Cd₁₋ᵧZnᵧTe",answers["czt_identity"])
        self.assertIn("InₓGa₁₋ₓAs",answers["ingaas_identity"])
        self.assertIn("D* = √(A_cm²)/NEP_density",answers["dstar"])
        self.assertIn("noise amplitude spectral density",answers["nasd"].lower())
        self.assertIn("photovoltaic photodiodes",answers["detector_types"].lower())

    def test_built_dataset_balance_and_no_exact_eval_leak(self):
        manifest=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        total=sum(manifest["categories"].values())
        self.assertGreaterEqual(manifest["categories"]["canonical_fact"],700)
        self.assertGreaterEqual(manifest["categories"]["quantitative_open"],2000)
        self.assertGreaterEqual(manifest["categories"]["diagnostic_reasoning"],100)
        self.assertLess(manifest["categories"]["out_of_scope"]/total,0.02)
        self.assertFalse(manifest["private_benchmark_imported"])
        self.assertFalse(manifest["exact_user_stress_prompts_imported"])
        prompts=set()
        for split in ("train","valid"):
            rows=[json.loads(x) for x in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines()]
            for row in rows:
                q=row["messages"][1]["content"]
                self.assertNotIn(q,prompts)
                prompts.add(q)
        held=set()
        for p in [ROOT/"evals/firm_specialist_scope_v1.jsonl",ROOT/"evals/firm_stress_v1.jsonl"]:
            if not p.exists():continue
            for line in p.read_text().splitlines():
                o=json.loads(line)
                if o.get("prompt"):held.add(o["prompt"])
        self.assertFalse(prompts&held)

    def test_hansen_exact_user_stress_pair_not_memorized(self):
        text=(DATA_DEFAULT/"train.jsonl").read_text()+(DATA_DEFAULT/"valid.jsonl").read_text()
        self.assertNotIn("8.1 µm at 300 K",text)
        self.assertNotIn("8.1um MCT at 300 K",text)

    def test_v1_known_false_strings_absent(self):
        text=((DATA_DEFAULT/"train.jsonl").read_text()+(DATA_DEFAULT/"valid.jsonl").read_text()).lower()
        for bad in ("nasdaq","bohn","haacke","in0.53ga0.47p","stainless steel",
                    "jones is s/cm/v","d*=lambda","d* = lambda","copper zinc telluride","cdznmct"):
            self.assertNotIn(bad,text)

    def test_provisional_gate_and_v2_hyperparameters(self):
        args=get_args(["--out","/tmp/firm4b-v2-test","--dry-run"])
        with self.assertRaises(PermissionError):verify_data(args)
        args.allow_provisional_data=True
        m,rows=verify_data(args)
        self.assertEqual(m["schema_version"],"2.0")
        self.assertEqual(len(rows["train"]),m["rows"]["train"])
        self.assertEqual(len(rows["valid"]),m["rows"]["valid"])
        self.assertEqual(args.learning_rate,1.5e-5)
        self.assertEqual(args.steps,1000)

    def test_dry_run_pins_same_base_and_rank16(self):
        import io
        from contextlib import redirect_stdout
        args=get_args(["--out","/tmp/firm4b-v2-test","--allow-provisional-data","--dry-run"])
        buf=io.StringIO()
        with redirect_stdout(buf):run(args)
        m=json.loads(buf.getvalue())
        self.assertEqual(m["model"],"Qwen/Qwen3.5-4B")
        self.assertEqual(m["revision"],QWEN35_4B_REVISION)
        self.assertEqual(m["rank"],16)
        self.assertEqual(m["alpha"],32)
        self.assertEqual(m["learning_rate"],1.5e-5)

if __name__=="__main__":unittest.main()
