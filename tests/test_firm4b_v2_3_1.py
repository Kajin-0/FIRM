"""Integrity gates for FIRM-4B v2.3.1 natural-prompt calibration."""
import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm4b_v2_3_1 import SYSTEM
from train_firm4b_v2_3_1 import get_args,verify_data,run,DATA_DEFAULT,_valid_roles
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

class Firm4BV231Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]
        for split in ("train","valid"):
            cls.rows += [json.loads(x) for x in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines() if x.strip()]
        cls.manifest=json.loads((DATA_DEFAULT/"manifest.json").read_text())

    def test_manifest_rebalanced(self):
        m=self.manifest
        self.assertEqual(m["schema_version"],"2.3.1")
        self.assertEqual(m["rows"]["train"],5489)
        self.assertEqual(m["rows"]["valid"],477)
        self.assertEqual(m["categories"]["canonical_fact"],1570)
        self.assertGreaterEqual(m["categories"]["quantitative_open"],1590)
        self.assertGreaterEqual(m["categories"]["natural_depth"],1000)
        self.assertGreaterEqual(m["categories"]["natural_depth_thinking"],240)
        self.assertGreaterEqual(m["categories"]["multiturn_depth"],220)
        self.assertGreaterEqual(m["calibration_fraction"],0.25)
        self.assertGreaterEqual(m["answer_stats"]["p75_words"],100)
        self.assertGreaterEqual(m["calibration_answer_stats"]["median_words"],110)
        self.assertFalse(m["private_benchmark_imported"])
        self.assertFalse(m["exact_user_stress_prompts_imported"])

    def test_system_defaults_to_complete_answers(self):
        s=SYSTEM.lower()
        for phrase in ("short user wording does not imply a short answer",
                       "60–120 words","120–300 word","follow-up asks for more",
                       "keep internal reasoning brief"):
            self.assertIn(phrase,s)

    def test_all_conversation_roles_and_multiturn(self):
        multi=[r for r in self.rows if r["category"]=="multiturn_depth"]
        self.assertGreaterEqual(len(multi),220)
        for r in self.rows:
            self.assertTrue(_valid_roles(r["messages"]),r["id"])
        for r in multi[:20]:
            self.assertEqual([m["role"] for m in r["messages"]],
                             ["system","user","assistant","user","assistant"])

    def test_natural_prompts_are_not_mostly_meta_depth_cues(self):
        rr=[r for r in self.rows if r["category"]=="natural_depth"]
        self.assertGreaterEqual(len(rr),1000)
        banned=("give a complete technical","technical deep dive","provide a complete specialist",
                "answer with enough detail","do not stop at taxonomy")
        bad=sum(any(x in r["messages"][1]["content"].lower() for x in banned) for r in rr)
        self.assertLess(bad,0.10*len(rr))

    def test_precision_targets_cover_observed_failures(self):
        targets="\n".join(r["messages"][-1].get("content","") for r in self.rows).lower()
        for term in ("thallium bromoiodide","tlbr","tli","kovar","cold-rolled",
                     "direct-bandgap","both direct-gap","in0.53ga0.47as","glass-to-metal"):
            self.assertIn(term,targets)
        for wrong in ("krs-5 is a germanium-arsenic","krs-5 is thallium antimonide",
                      "hgcdte is a direct-gap iii–vi"):
            self.assertNotIn(wrong,targets)

    def test_user_regressions_remain_held_out_across_all_turns(self):
        hashes=set()
        for r in self.rows:
            for m in r["messages"]:
                if m.get("role")=="user":
                    hashes.add(hashlib.sha256(m["content"].lower().encode()).hexdigest())
        self.assertFalse(hashes & USER_REGRESSION_HASHES)

    def test_no_public_or_shadow_gauntlet_leakage(self):
        user_turns={m["content"] for r in self.rows for m in r["messages"] if m.get("role")=="user"}
        for name in ("firm_v22_gauntlet_v1.jsonl","firm_v23_depth_gauntlet_v1.jsonl",
                     "firm_v231_shadow_gauntlet_v1.jsonl"):
            held={json.loads(x)["prompt"] for x in (ROOT/"evals"/name).read_text().splitlines() if x.strip()}
            self.assertFalse(user_turns & held)

    def test_thinking_targets_short_reasoning_long_final(self):
        rr=[r for r in self.rows if r["category"]=="natural_depth_thinking"]
        self.assertGreaterEqual(len(rr),240)
        for r in rr:
            a=r["messages"][-1]
            self.assertTrue(a.get("reasoning_content","").strip())
            self.assertLess(len(a["reasoning_content"].split()),70)
            self.assertGreaterEqual(len(a["content"].split()),80)

    def test_balanced_canonical_variants(self):
        rr=[r for r in self.rows if r["category"]=="canonical_fact"]
        counts={}
        for r in rr:
            fid=r["metadata"]["fact_id"];counts[fid]=counts.get(fid,0)+1
            self.assertLess(r["metadata"]["variant"],10)
        self.assertEqual(len(counts),157)
        self.assertEqual(set(counts.values()),{10})

    def test_hashes_verify_and_one_pass_dry_run(self):
        m=self.manifest
        for split in ("train","valid"):
            p=DATA_DEFAULT/(split+".jsonl")
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m[split+"_sha256"])
        args=get_args(["--out","/tmp/firm-v231-test","--dry-run"])
        with self.assertRaises(PermissionError): verify_data(args)
        args.allow_provisional_data=True
        mm,rows=verify_data(args)
        self.assertEqual(len(rows["train"]),5489)
        import io
        from contextlib import redirect_stdout
        buf=io.StringIO()
        with redirect_stdout(buf): run(args)
        meta=json.loads(buf.getvalue())
        self.assertEqual(meta["model"],"Qwen/Qwen3.5-4B")
        self.assertEqual(meta["revision"],QWEN35_4B_REVISION)
        self.assertEqual(meta["training_steps"],1373)
        self.assertEqual(meta["rank"],16)
        self.assertEqual(meta["learning_rate"],1.5e-5)
        eff=meta["training_steps"]*meta["effective_batch_size"]
        self.assertGreaterEqual(eff,meta["train_examples"])
        self.assertLess(eff,1.002*meta["train_examples"])

if __name__=="__main__":unittest.main()
