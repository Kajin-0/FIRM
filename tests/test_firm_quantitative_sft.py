"""Invariants for original FIRM quantitative synthesis and training."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_firm_quantitative_sft import FAMILIES,case,generate
from train_firm_quant_sft import get_args,verify_data

class QuantitativeSFTTests(unittest.TestCase):
    def test_each_formula_independent_invariant_and_units(self):
        for family in FAMILIES:
            for idx in range(30):
                prompt,val,unit,key,notes,check=case(family,random.Random(f"science-test-{family}-{idx}"))
                self.assertTrue(check(val),(family,idx,val))
                self.assertTrue(prompt and unit and key and notes)
                self.assertGreaterEqual(len(notes),35)

    def test_full_generation_has_disjoint_splits_and_valid_oracles(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            m=generate(p)
            args=get_args(["--data",str(p),"--out",str(p/"out"),"--dry-run"])
            _,rows=verify_data(args)
            self.assertEqual((len(rows["train"]),len(rows["valid"])),(2640,264))
            self.assertEqual(len({x["messages"][1]["content"] for x in rows["train"]}),2640)
            self.assertEqual(len({x["messages"][1]["content"] for x in rows["valid"]}),264)
            self.assertEqual(m["family_count"],22)
            for split in ("train","valid"):
                for r in rows[split]:
                    q=json.loads(r["messages"][2]["content"])
                    self.assertEqual(set(q),{"answer","quantities"})

    def test_determinism_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            generate(p)
            train=p/"train.jsonl"
            first=train.read_bytes()
            generate(p)
            self.assertEqual(first,train.read_bytes())
            args=get_args(["--data",str(p),"--out",str(p/"out"),"--dry-run"])
            train.write_text(train.read_text()+"\n",encoding="utf8")
            with self.assertRaisesRegex(ValueError,"Corrupt"):
                verify_data(args)

    def test_bounded_training_budget(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);generate(p)
            for steps in (0,2,99,701,10000):
                args=get_args(["--data",str(p),"--out",str(p/"out"),"--steps",str(steps),"--dry-run"])
                with self.assertRaisesRegex(ValueError,"Unexpected experiment budget"):
                    verify_data(args)

if __name__=="__main__":unittest.main()
