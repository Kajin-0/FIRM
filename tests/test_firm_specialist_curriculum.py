"""Specialist conceptual curriculum integrity and scope-data protections."""
import json
import tempfile
import unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_firm_specialist_curriculum import DOMAIN_CARDS,OUTSIDE_TOPICS,build

class SpecialistCurriculumTests(unittest.TestCase):
    def test_curated_card_structure_and_mix(self):
        self.assertGreaterEqual(len(DOMAIN_CARDS),50)
        self.assertGreaterEqual(len(OUTSIDE_TOPICS),40)
        ids=set()
        for id_,q,a in DOMAIN_CARDS:
            self.assertNotIn(id_,ids)
            ids.add(id_)
            self.assertGreater(len(q),25)
            self.assertGreater(len(a),110)
            self.assertNotIn("http://",a)
            self.assertNotIn("https://",a)
        self.assertEqual(len({q for _,q in OUTSIDE_TOPICS}),len(OUTSIDE_TOPICS))
    def test_generation_and_no_scope_eval_prompt_collisions(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"concepts"
            m=build(path)
            assert m["outputs"]["train"]["rows"]==657
            assert m["outputs"]["valid"]["rows"]==101
            assert m["training_status"].startswith("BLOCKED")
            splits={}
            eval_prompts={json.loads(s)["prompt"] for s in (ROOT/"evals/firm_specialist_scope_v1.jsonl").read_text().splitlines()}
            for split in ("train","valid"):
                records=[json.loads(x) for x in (path/(split+".jsonl")).read_text().splitlines()]
                prompts={x["messages"][0]["content"] for x in records}
                self.assertEqual(len(prompts),len(records))
                self.assertFalse(prompts & eval_prompts)
                for record in records:
                    self.assertFalse(record["training_eligible"])
                    self.assertEqual([z["role"] for z in record["messages"]],["user","assistant"])
                    self.assertIn(record["provenance"]["training_rights"],["original"])
                    self.assertIn(record["category"],("domain_conceptual","out_of_scope"))
                    self.assertIn(record["provenance"]["review_status"],("awaiting_independent_expert_audit","scope_label_needs_review"))
                splits[split]=prompts
            self.assertFalse(splits["train"]&splits["valid"])
    def test_no_benchmark_data_imports(self):
        import inspect
        from build_firm_specialist_curriculum import build
        src=inspect.getsource(build)
        self.assertNotIn("firm_large_quant",src)
        self.assertNotIn("firm_specialist_scope_v1.jsonl",src)
        self.assertNotIn("firm_science_dev_v2",src)
    def test_refusal_answers_consistent_and_in_domain_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"curriculum"
            build(p)
            for split in ("train","valid"):
                rows=[json.loads(x) for x in (p/(split+".jsonl")).read_text().splitlines()]
                for r in rows:
                    a=r["messages"][1]["content"]
                    if r["category"]=="out_of_scope":
                        self.assertIn("outside my scope",a)
                    else:
                        self.assertNotIn("outside my scope",a)

if __name__=="__main__":unittest.main()
