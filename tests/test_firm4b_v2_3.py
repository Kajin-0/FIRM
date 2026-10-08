"""Integrity gates for FIRM-4B v2.3 adaptive-depth curriculum/trainer."""
import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm4b_v2_3 import SYSTEM
from train_firm4b_v2_3 import get_args,verify_data,run,DATA_DEFAULT
from firm_model_profiles import QWEN35_4B_REVISION

USER_REGRESSION_HASHES={
"3756d429ab1009e4fc880ae5a7f3ce1a7d9f8ff50188328da1025784e74c0fe6",
"2dbc34d375d014e3c34dae811b4a248668ae520cf967bf504ed51eedf874e8ff",
"8eaaf9d51895167213c676a349d500bd1a813515a5464c5f39990346ef0a8da5",
"fa197d421d067f036e2dbdc7ef69bb0c2409821e4b809db7d1b65970633dc0a7",
"ab136b5de9c4ebfe4553405c77795241edd3b4a5552d63362d13ad0038af3675",
"ab5ebd9d7050ddbe061c45a9972e141fc65daf9058ecc8a55ec308cbe56a448f",
"3a9b3c68ff5de6dff34b55bf0fcfec55f22152c6b6733c69358ed605dff3688f",
}

class Firm4BV23Tests(unittest.TestCase):
    def _rows(self):
        out=[]
        for split in ("train","valid"):
            out += [json.loads(x) for x in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines() if x.strip()]
        return out

    def test_manifest_depth_and_counts(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        self.assertEqual(m["schema_version"],"2.3")
        self.assertGreaterEqual(m["rows"]["train"],5600)
        self.assertGreaterEqual(m["categories"]["depth_complete"],380)
        self.assertGreaterEqual(m["categories"]["precision_correction"],60)
        self.assertGreaterEqual(m["categories"]["reasoning_depth"],60)
        self.assertGreaterEqual(m["depth_answer_stats"]["median_words"],110)
        self.assertGreaterEqual(m["answer_stats"]["p95_words"],100)
        self.assertFalse(m["private_benchmark_imported"])
        self.assertFalse(m["exact_user_stress_prompts_imported"])

    def test_response_policy_requires_adaptive_completeness(self):
        s=SYSTEM.lower()
        for phrase in ("match answer depth","decision-relevant","not incomplete","expand materially","validation steps"):
            self.assertIn(phrase,s)
        self.assertNotIn("directly, accurately, and concisely",s)

    def test_precision_targets_cover_failures(self):
        targets="\n".join(r["messages"][-1].get("content","") for r in self._rows()).lower()
        for term in ("krs-5","thallium bromoiodide","kovar","cold-rolled","to-66",
                     "direct-bandgap","in0.53ga0.47as","glass-to-metal"):
            self.assertIn(term,targets)
        self.assertIn("both direct-gap",targets)
        self.assertNotIn("krs-5 is a germanium-arsenic",targets)

    def test_user_regressions_remain_held_out(self):
        hashes={hashlib.sha256(r["messages"][1]["content"].lower().encode()).hexdigest() for r in self._rows()}
        self.assertFalse(hashes & USER_REGRESSION_HASHES)

    def test_no_public_gauntlet_leakage(self):
        train_valid={r["messages"][1]["content"] for r in self._rows()}
        for name in ("firm_v22_gauntlet_v1.jsonl","firm_v23_depth_gauntlet_v1.jsonl"):
            held={json.loads(x)["prompt"] for x in (ROOT/"evals"/name).read_text().splitlines() if x.strip()}
            self.assertFalse(train_valid & held)

    def test_long_answers_are_real_not_empty_padding(self):
        rows=[r for r in self._rows() if r["category"]=="depth_complete"]
        self.assertGreaterEqual(len(rows),380)
        for r in rows:
            a=r["messages"][-1]["content"]
            self.assertGreaterEqual(len(a.split()),80)
            self.assertLess(len(a.split()),220)

    def test_reasoning_stays_short_while_answer_can_be_long(self):
        rows=[r for r in self._rows() if r["category"]=="reasoning_depth"]
        self.assertGreaterEqual(len(rows),60)
        for r in rows:
            a=r["messages"][-1]
            self.assertTrue(a.get("reasoning_content","").strip())
            self.assertLess(len(a["reasoning_content"].split()),70)
            self.assertGreater(len(a["content"].split()),80)

    def test_hashes_and_dry_run(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        for split in ("train","valid"):
            p=DATA_DEFAULT/(split+".jsonl")
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m[split+"_sha256"])
        args=get_args(["--out","/tmp/firm-v23-test","--dry-run"])
        with self.assertRaises(PermissionError): verify_data(args)
        args.allow_provisional_data=True
        mm,rows=verify_data(args)
        self.assertEqual(len(rows["train"]),m["rows"]["train"])
        import io
        from contextlib import redirect_stdout
        buf=io.StringIO()
        with redirect_stdout(buf): run(args)
        meta=json.loads(buf.getvalue())
        self.assertEqual(meta["model"],"Qwen/Qwen3.5-4B")
        self.assertEqual(meta["revision"],QWEN35_4B_REVISION)
        self.assertEqual(meta["training_steps"],1413)
        self.assertEqual(meta["rank"],16)
        self.assertEqual(meta["learning_rate"],1.5e-5)
        eff=meta["training_steps"]*meta["effective_batch_size"]
        self.assertGreaterEqual(eff,meta["train_examples"])
        self.assertLess(eff,1.01*meta["train_examples"])

if __name__=="__main__": unittest.main()
