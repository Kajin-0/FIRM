"""Source provenance registry must never silently authorize scientific-text training."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class SourceRightsTests(unittest.TestCase):
    def test_scientific_source_candidates_are_bibliography_only(self):
        p=ROOT/"data/firm_scientific_source_candidates_v1.jsonl"
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        self.assertGreaterEqual(len(rows),7)
        ids=set()
        for row in rows:
            self.assertNotIn(row["key"],ids)
            ids.add(row["key"])
            self.assertTrue(row["url"].startswith("https://"))
            self.assertGreater(len(row["claims_to_independently_audit"]),0)
            self.assertEqual(row["training_status"],"blocked_pending_rights_and_independent_claim_review")
            self.assertEqual(row["source_role"],"candidate_bibliography_not_training_text")
            self.assertNotIn("full_text",row)
            self.assertTrue(any(term in row["rights_status"].lower() for term in ("unverified","not_verified")))
    def test_no_source_candidates_in_sft_corpus(self):
        import hashlib
        p=ROOT/"data/processed/firm3_synthetic_quant_v1"
        candidates=[json.loads(x) for x in (ROOT/"data/firm_scientific_source_candidates_v1.jsonl").read_text().splitlines()]
        known_urls={c["url"] for c in candidates}
        for split in ("train.jsonl","valid.jsonl"):
            records=(p/split).read_text()
            self.assertTrue(all(url not in records for url in known_urls))

if __name__=="__main__":unittest.main()
