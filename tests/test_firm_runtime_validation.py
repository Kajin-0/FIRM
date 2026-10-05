"""Runtime safety gates do not import GPU libraries or load weights."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from firm_data import sha256
from firm_model_profiles import modern_targets,validate_profile,QWEN35_REVISION
from run_firm_baseline import messages_for,ollama_metadata
from firm_run_context import repository_revision
from train_firm_qlora import require_reviewed_inputs
from build_firm_science_dev import development_items

class RuntimeValidationTests(unittest.TestCase):
    def test_modern_targets_exclude_visual_and_hybrid_state_and_E2_is_gated(self):
        names=['model.visual.blocks.0.mlp.gate_proj','model.language_model.layers.0.mlp.gate_proj',
               'model.language_model.layers.0.linear_attn.in_proj_a','model.language_model.layers.3.self_attn.q_proj','lm_head']
        self.assertEqual(modern_targets(names),[names[1]])
        args=SimpleNamespace(model_profile='qwen35_text_bf16',model='Qwen/Qwen3.5-9B',model_revision=QWEN35_REVISION,
                             quantization='none',lora_dropout=0,max_steps=2,max_train_examples=8)
        validate_profile(args)
        args.quantization='nf4'
        with self.assertRaises(ValueError):validate_profile(args)
        args.quantization='none';args.max_steps=100
        with self.assertRaises(ValueError):validate_profile(args)

    def test_compact_protocol_withholds_gold_and_cloud_models_rejected(self):
        case=development_items()[0]
        original=messages_for(case,'scientific-compact-v2')
        altered=copy.deepcopy(case)
        for q in altered['grading']['quantities'].values():q.update(value=98765,unit='madeup')
        altered['grading']['manual_rubric']=['SECRET GOLD']
        self.assertEqual(original,messages_for(altered,'scientific-compact-v2'))
        import io
        reply=io.BytesIO(json.dumps({'models':[{'name':'cloud:latest','digest':'x','remote_host':'ollama.com'}]}).encode())
        with patch('run_firm_baseline.urllib.request.urlopen',return_value=reply):
            with self.assertRaises(ValueError):ollama_metadata('http://127.0.0.1:11434','cloud:latest','x')

    def test_extracted_package_rejects_changed_payload_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);payload=root/'payload.txt';payload.write_text('trusted fixture')
            (root/'firm_gpu_package_manifest.json').write_text(json.dumps({'git_sha':'0'*40,'assets':[{'path':'payload.txt','sha256':sha256(payload)}]}))
            self.assertEqual(repository_revision(root),'0'*40)
            payload.write_text('changed fixture')
            with self.assertRaises(ValueError):repository_revision(root)

    def test_actual_training_requires_reviewed_eligible_rows(self):
        root=ROOT/'data/processed/firm3_reviewed_seed_v3'
        args=SimpleNamespace(train=root/'train.jsonl',valid=root/'valid.jsonl',max_train_examples=8)
        require_reviewed_inputs(args)
        args.train=root/'quarantine.jsonl'
        with self.assertRaises(ValueError):require_reviewed_inputs(args)
        args.train=ROOT/'data/processed/firm3_candidate_2026-10-05_v2/train.jsonl'
        if args.train.exists():
            with self.assertRaises(ValueError):require_reviewed_inputs(args)

if __name__=="__main__":unittest.main()
