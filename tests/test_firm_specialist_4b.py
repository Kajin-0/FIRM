"""Compact specialist 4B profile/data permission and source integrity gates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from firm_model_profiles import QWEN35_4B_REVISION,modern_targets,validate_profile
from assemble_firm_specialist_dataset import summary,build,ROOT as DATA_ROOT
from review_firm_specialist_cards import inspect
from train_firm_specialist_4b import get_args,verify_data

class Compact4BTests(unittest.TestCase):
    def test_pinned_4b_profile_and_forbidden_quantization(self):
        self.assertEqual(len(QWEN35_4B_REVISION),40)
        args=SimpleNamespace(model="Qwen/Qwen3.5-4B",model_revision=QWEN35_4B_REVISION,
                             model_profile="qwen35_4b_text_bf16",quantization="none",lora_dropout=0,
                             max_steps=1100,max_train_examples=4311)
        validate_profile(args)
        args.quantization="nf4"
        with self.assertRaisesRegex(ValueError,"NF4"):
            validate_profile(args)
        args.quantization="none";args.model_revision="main"
        with self.assertRaisesRegex(ValueError,"revision"):
            validate_profile(args)
    def test_language_only_32_layers_96_mlp_targets(self):
        names=[f"model.language_model.layers.{i}.mlp.{projection}"
               for i in range(32) for projection in ("gate_proj","up_proj","down_proj")]
        names.extend(["model.visual.blocks.0.mlp.up_proj","model.language_model.layers.3.self_attn.q_proj"])
        self.assertEqual(len(modern_targets(names)),96)
    def test_review_ledger_all_cards_still_blocked(self):
        state=summary()
        self.assertEqual(state["review"]["total"],101)
        self.assertEqual(state["review"]["blocked"],101)
        self.assertFalse(state["training_permission"])
        review=inspect()
        self.assertEqual(len(review["approved_keys"]),0)
    def test_training_requires_explicit_provisional_permission(self):
        args=get_args(["--data",str(DATA_ROOT/"data/processed/firm3_specialist_4b_mix_v1"),
                       "--out","/tmp/firm-four-billion-training"])
        with self.assertRaisesRegex(PermissionError,"review"):
            verify_data(args)
        args.allow_provisional_data=True
        manifest,rows=verify_data(args)
        self.assertEqual((len(rows["train"]),len(rows["valid"])),(4311,365))
        self.assertEqual(manifest["review_status"],"PROVISIONAL_UNREVIEWED_RESEARCH_ONLY")
    def test_exact_four_billion_evaluator_pins_checkpoint(self):
        from eval_firm_quant_sft import get_args as eval_args, run as eval_run
        from contextlib import redirect_stdout
        import io
        a=eval_args(["--model","Qwen/Qwen3.5-4B","--out","/tmp/firm4b-dryrun-score","--dry-run"])
        text=io.StringIO()
        with redirect_stdout(text):eval_run(a)
        m=json.loads(text.getvalue())
        self.assertEqual(m["model"],"Qwen/Qwen3.5-4B")
        self.assertEqual(m["revision"],QWEN35_4B_REVISION)
        self.assertEqual(m["mode"],"base")

    def test_assembly_requires_review_or_explicit_provisional_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"refused"
            with self.assertRaises(PermissionError):build(p)
            self.assertFalse(p.exists())
            m=build(p,provisional=True)
            self.assertEqual(m["outputs"]["train"]["rows"],4311)
            self.assertEqual(m["outputs"]["valid"]["rows"],365)
            self.assertIn("PROVISIONAL",m["review_status"])

if __name__=="__main__":unittest.main()
