"""Integrity gates for expanded FIRM-4B v2.1 curriculum/trainer."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm4b_v2_expanded import (
    EXTRA_FACTS,EXTRA_DIAGNOSTICS,EXTRA_TAXONOMY,CONTRASTIVE,build
)
from train_firm4b_v2_1 import get_args,verify_data,run,DATA_DEFAULT
from firm_model_profiles import QWEN35_4B_REVISION

class Firm4BV21Tests(unittest.TestCase):
    def test_expansion_is_conceptual_not_only_prompt_wrappers(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        self.assertEqual(m["schema_version"],"2.1")
        self.assertGreaterEqual(m["foundational_fact_cards"],150)
        self.assertGreaterEqual(m["diagnostic_unique_targets"],50)
        self.assertGreaterEqual(m["taxonomy_unique_targets"],16)
        self.assertGreaterEqual(m["contrastive_unique_targets"],20)
        self.assertGreaterEqual(len(EXTRA_FACTS),70)
        self.assertGreaterEqual(len(EXTRA_DIAGNOSTICS),30)
        self.assertGreaterEqual(len(EXTRA_TAXONOMY),10)
        self.assertGreaterEqual(len(CONTRASTIVE),20)

    def test_balance_and_unique_answers(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        total=sum(m["categories"].values())
        self.assertGreaterEqual(m["categories"]["canonical_fact"],2400)
        self.assertGreaterEqual(m["categories"]["quantitative_open"],2300)
        self.assertGreaterEqual(m["categories"]["diagnostic_reasoning"],300)
        self.assertGreaterEqual(m["categories"]["contrastive_correction"],90)
        self.assertLess(m["categories"]["out_of_scope"]/total,0.01)
        rows=[]
        for split in ("train","valid"):
            rows += [json.loads(x) for x in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines()]
        self.assertEqual(len(rows),len({r["messages"][1]["content"] for r in rows}))
        self.assertGreaterEqual(len({r["messages"][2]["content"] for r in rows}),2500)

    def test_v1_failure_modes_directly_covered_without_exact_stress_import(self):
        rows=[]
        for split in ("train","valid"):
            rows += [json.loads(x) for x in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines()]
        targets="\n".join(r["messages"][2]["content"] for r in rows).lower()
        must_have=[
            "mercury cadmium telluride",
            "cadmium zinc telluride",
            "indium gallium arsenide",
            "noise amplitude spectral density",
            "cm·√hz/w",
            "in0.53ga0.47as",
            "d*=√a",
        ]
        for good in must_have:self.assertIn(good,targets)
        for bad in ("nasdaq","bohn","haacke","in0.53ga0.47p","jones is s/cm/v",
                    "copper zinc telluride","cdznmct"):
            self.assertNotIn(bad,targets)
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        self.assertFalse(m["private_benchmark_imported"])
        self.assertFalse(m["exact_user_stress_prompts_imported"])

    def test_no_exact_public_eval_prompt_leakage(self):
        train_valid=set()
        for split in ("train","valid"):
            for line in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines():
                train_valid.add(json.loads(line)["messages"][1]["content"])
        held=set()
        for p in (ROOT/"evals/firm_specialist_scope_v1.jsonl",ROOT/"evals/firm_stress_v1.jsonl"):
            if not p.exists():continue
            for line in p.read_text().splitlines():
                o=json.loads(line)
                if o.get("prompt"):held.add(o["prompt"])
        self.assertFalse(train_valid & held)

    def test_manifest_hashes_and_provisional_gate(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        for split in ("train","valid"):
            data=(DATA_DEFAULT/(split+".jsonl")).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),m[split+"_sha256"])
        args=get_args(["--out","/tmp/firm-v21-test","--dry-run"])
        with self.assertRaises(PermissionError):verify_data(args)
        args.allow_provisional_data=True
        mm,rows=verify_data(args)
        self.assertEqual(len(rows["train"]),4973)
        self.assertEqual(len(rows["valid"]),432)

    def test_dry_run_exact_base_and_near_one_pass(self):
        import io
        from contextlib import redirect_stdout
        args=get_args(["--out","/tmp/firm-v21-test","--allow-provisional-data","--dry-run"])
        buf=io.StringIO()
        with redirect_stdout(buf):run(args)
        m=json.loads(buf.getvalue())
        self.assertEqual(m["model"],"Qwen/Qwen3.5-4B")
        self.assertEqual(m["revision"],QWEN35_4B_REVISION)
        self.assertEqual(m["rank"],16)
        self.assertEqual(m["alpha"],32)
        self.assertEqual(m["learning_rate"],1.5e-5)
        self.assertEqual(m["training_steps"],1250)
        effective_examples=m["training_steps"]*m["effective_batch_size"]
        self.assertGreaterEqual(effective_examples,m["train_examples"])
        self.assertLess(effective_examples,1.05*m["train_examples"])

if __name__=="__main__":unittest.main()
