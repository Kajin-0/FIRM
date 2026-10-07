"""Integrity gates for FIRM-4B v2.2 curriculum/trainer."""
import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm4b_v2_2 import PROCESS_FACTS,PROCESS_DIAGNOSTICS,CONTRASTIVE,SYSTEM
from train_firm4b_v2_2 import get_args,verify_data,run,DATA_DEFAULT
from firm_model_profiles import QWEN35_4B_REVISION

USER_STRESS_HASHES={
    "9de954c21367385ee7ca841ac07fe1d117784316a7910aaa0432651e8ffbd049",
    "a50849bd71072e3110ab23e328605cfd81af07a55434bb36e004526badfa9791",
    "ed48c716c74c4b5892e277d0dfd824e6032890415eb0e08e37c7c6d0c139d358",
}

class Firm4BV22Tests(unittest.TestCase):
    def _rows(self):
        out=[]
        for split in ("train","valid"):
            out += [json.loads(x) for x in (DATA_DEFAULT/(split+".jsonl")).read_text().splitlines() if x.strip()]
        return out

    def test_manifest_and_counts(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        self.assertEqual(m["schema_version"],"2.2")
        self.assertGreaterEqual(m["rows"]["train"],5000)
        self.assertGreaterEqual(m["process_fact_targets"],10)
        self.assertGreaterEqual(m["process_diagnostic_targets"],10)
        self.assertGreaterEqual(m["categories"]["process_engineering"],50)
        self.assertGreaterEqual(m["categories"]["reasoning_trace"],60)
        self.assertEqual(m["thinking_trace_targets"],12)
        self.assertFalse(m["private_benchmark_imported"])
        self.assertFalse(m["exact_user_stress_prompts_imported"])

    def test_response_policy_is_substantive(self):
        s=SYSTEM.lower()
        for phrase in ("enough substance","concrete options","tradeoffs","failure modes","generic caveats","correct false premises"):
            self.assertIn(phrase,s)

    def test_process_targets_cover_new_failure_classes(self):
        targets="\n".join(r["messages"][2]["content"] for r in self._rows()).lower()
        for term in ("bond-line","cte","outgassing","bromine","mercury","chemical potential",
                     "photoconductivity","in0.53ga0.47as","thermal cycling"):
            self.assertIn(term,targets)
        self.assertIn("willoughby smith",targets)
        self.assertIn("english electrical engineer",targets)
        self.assertIn("ii-vi",targets.replace("ii–vi","ii-vi"))

    def test_exact_user_stress_prompts_are_not_training_data(self):
        import hashlib
        prompt_hashes={hashlib.sha256(r["messages"][1]["content"].lower().encode()).hexdigest() for r in self._rows()}
        self.assertFalse(prompt_hashes & USER_STRESS_HASHES)

    def test_no_empty_or_duplicate_prompts(self):
        rows=self._rows()
        prompts=[r["messages"][1]["content"] for r in rows]
        self.assertEqual(len(prompts),len(set(prompts)))
        self.assertTrue(all(r["messages"][2]["content"].strip() for r in rows))

    def test_no_exact_v22_gauntlet_prompt_leakage(self):
        train_valid={r["messages"][1]["content"] for r in self._rows()}
        held={json.loads(x)["prompt"] for x in (ROOT/"evals/firm_v22_gauntlet_v1.jsonl").read_text().splitlines() if x.strip()}
        self.assertFalse(train_valid & held)

    def test_reasoning_traces_are_short_and_have_final_answers(self):
        rows=[r for r in self._rows() if r["category"]=="reasoning_trace"]
        self.assertGreaterEqual(len(rows),60)
        for r in rows:
            a=r["messages"][2]
            self.assertTrue(a.get("reasoning_content","").strip())
            self.assertTrue(a.get("content","").strip())
            self.assertLess(len(a["reasoning_content"].split()),90)

    def test_hashes_and_dry_run(self):
        m=json.loads((DATA_DEFAULT/"manifest.json").read_text())
        for split in ("train","valid"):
            p=DATA_DEFAULT/(split+".jsonl")
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m[split+"_sha256"])
        args=get_args(["--out","/tmp/firm-v22-test","--dry-run"])
        with self.assertRaises(PermissionError):verify_data(args)
        args.allow_provisional_data=True
        mm,rows=verify_data(args)
        self.assertEqual(len(rows["train"]),m["rows"]["train"])
        import io
        from contextlib import redirect_stdout
        buf=io.StringIO()
        with redirect_stdout(buf):run(args)
        meta=json.loads(buf.getvalue())
        self.assertEqual(meta["model"],"Qwen/Qwen3.5-4B")
        self.assertEqual(meta["revision"],QWEN35_4B_REVISION)
        self.assertEqual(meta["training_steps"],1290)
        self.assertEqual(meta["rank"],16)
        self.assertEqual(meta["learning_rate"],1.5e-5)
        eff=meta["training_steps"]*meta["effective_batch_size"]
        self.assertGreaterEqual(eff,meta["train_examples"])
        self.assertLess(eff,1.01*meta["train_examples"])

if __name__=="__main__": unittest.main()
