"""Lockdown tests for the 11,500-question private FIRM evaluation bank."""
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm_large_benchmark import (
    TRAIN_FAMILIES,UNSEEN_FAMILIES,unseen_case,build
)
from score_firm_large_quant import grade,score,wilson

class LockedBenchmarkTests(unittest.TestCase):
    def test_independent_unseen_formula_invariants(self):
        self.assertEqual(len(UNSEEN_FAMILIES),15)
        self.assertEqual(len(TRAIN_FAMILIES),22)
        self.assertEqual(set(TRAIN_FAMILIES)&set(UNSEEN_FAMILIES),set())
        for family in UNSEEN_FAMILIES:
            for i in range(80):
                q,key,value,unit,explanation,checker=unseen_case(
                    family,random.Random(f"invariant-{family}-{i}"))
                self.assertTrue(checker(value),(family,i))
                self.assertTrue(q and key and unit and explanation)
    def test_benchmark_generated_private_seed_and_train_exclusion(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)
            m=build(dest,2,3,seed="private-fixture-seed-0123456789abcdef0123456789abcdef")
            self.assertNotIn("seed",m)
            self.assertEqual([x["rows"] for x in m["tiers"]],[44,45])
            manifest=(dest/"manifest.json").read_bytes()
            for asset in m["tiers"]:
                raw=(dest/asset["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),asset["sha256"])
            regenerated=build(dest,2,3,seed="private-fixture-seed-0123456789abcdef0123456789abcdef",force=True)
            self.assertEqual(manifest,(dest/"manifest.json").read_bytes())
            self.assertEqual(m,regenerated)
    def test_scoring_units_numerical_and_strict_json(self):
        gold={"grading":{"quantities":{"dstar":{"value":2.5e9,"unit":"Jones","rtol":.003,"atol":1e-10}}}}
        fine=json.dumps({"quantities":{"dstar":{"value":2.5e9,"unit":"Jones"}}})
        score1=grade(gold,{"response":fine,"finish_reason":"stop"})
        self.assertTrue(score1["joint_correct"])
        wrong=grade(gold,{"response":json.dumps({"quantities":{"dstar":{"value":2.5e9,"unit":"cm"}}})})
        self.assertTrue(wrong["number_correct"])
        self.assertFalse(wrong["joint_correct"])
        malformed=grade(gold,{"response":"the answer is 2.5e9 Jones"})
        self.assertFalse(malformed["json_valid"])
        self.assertFalse(malformed["number_correct"])
    def test_locked_bank_rejects_duplicate_and_partial(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t); m=build(dest,1,1,seed="test-secret-seed-0123456789abcdef0123456789abcdef")
            tier="known_family_parameter_holdout"
            first=json.loads((dest/(tier+".jsonl")).read_text().splitlines()[0])
            p=dest/"predictions.jsonl"
            p.write_text(json.dumps({"id":first["id"],"response":"{}"})+"\n")
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                score(dest,p,tier)
            summary=score(dest,p,tier,allow_partial=True)
            self.assertFalse(summary["completed"])
            self.assertLess(summary["coverage"],1)
            p.write_text(p.read_text()+p.read_text())
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                score(dest,p,tier,allow_partial=True)
    def test_confidence_interval(self):
        for good,n in [(0,100),(40,100),(100,100),(1,1)]:
            lo,hi=wilson(good,n)
            self.assertLessEqual(lo,good/n)
            self.assertGreaterEqual(hi,good/n)
            self.assertTrue(0<=lo<=hi<=1)

if __name__=="__main__":unittest.main()
