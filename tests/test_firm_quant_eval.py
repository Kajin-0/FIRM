"""Scoring and precision boundaries of FIRM quantitative evaluation."""
import json,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_firm_quant_sft import grade,get_args,run

class QuantitativeGradeTests(unittest.TestCase):
    def setUp(self):
        self.row={"id":"fixture","metadata":{"oracle":{"key":"specific_detectivity","unit":"Jones","value":1e10},"family":"detectivity"}}
    def test_numeric_and_units(self):
        good=json.dumps({"quantities":{"specific_detectivity":{"value":1.001e10,"unit":"Jones"}},"answer":"validated"})
        result=grade(self.row,good)
        self.assertTrue(result["correct_number"])
        self.assertTrue(result["correct_unit"])
        self.assertTrue(result["strict_json"])
    def test_wrong_units_do_not_pass(self):
        wrong=json.dumps({"quantities":{"specific_detectivity":{"value":1e10,"unit":"m sqrt(Hz)/W"}},"answer":"same number wrong output units"})
        result=grade(self.row,wrong)
        self.assertTrue(result["correct_number"])
        self.assertFalse(result["correct_unit"])
    def test_nonfinite_and_unstructured_fail(self):
        for answer in ("not JSON",json.dumps({"quantities":{"specific_detectivity":{"value":"NaN","unit":"Jones"}},"answer":"invalid"}),json.dumps({"answer":"not enough fields"})):
            result=grade(self.row,answer)
            self.assertFalse(result["correct_number"])
    def test_eval_preflight_does_not_create_output(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            args=get_args(["--out",root+"/unused","--dry-run"])
            run(args)
            self.assertFalse(Path(root,"unused").exists())

if __name__=="__main__":unittest.main()
