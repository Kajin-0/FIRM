"""Runtime safety gates do not import GPU libraries or load weights."""
import copy
import io
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
from run_firm_baseline import messages_for,ollama_metadata,prediction_record
from firm_run_context import repository_revision
from train_firm_qlora import require_reviewed_inputs
from build_firm_science_dev import development_items
from eval_firm_science import grade_quantity
import run_firm_baseline

class RuntimeValidationTests(unittest.TestCase):
    def test_observed_valid_unit_spellings_do_not_fail_or_change_dimensions(self):
        for canonical,alternative in [('Ohm','V/A'),('s','seconds'),('Jones','cm*Hz^{1/2}/W'),
                                       ('W/(m^2*sr*um)','W/(sr*m^2*um)')]:
            self.assertTrue(grade_quantity({'value':2,'unit':canonical,'rtol':0,'atol':0},
                                           {'value':2,'unit':alternative})['numerical_correct'])
        self.assertFalse(grade_quantity({'value':2,'unit':'V/W','rtol':0,'atol':0},
                                        {'value':2,'unit':'V/A'})['unit_correct'])

    def test_legacy_prose_has_no_structured_contract_and_raw_response_is_preserved(self):
        obj={'message':{'content':'A prose scientific answer.', 'thinking':'trace'},
             'done':True,'done_reason':'stop','eval_count':8,'prompt_eval_count':12}
        original=copy.deepcopy(obj)
        row=prediction_record({'id':'legacy'},obj,'ollama',0.5,'hash')
        self.assertNotIn('structured_parse_error',row)
        self.assertIsNone(row['generation_error'])
        self.assertEqual(row['reasoning_content'],'trace')
        self.assertEqual(obj,original)
        structured=prediction_record({'id':'numeric','grading':{}},obj,'ollama',0.5,'hash')
        self.assertTrue(structured['structured_parse_error'])

    def test_nonterminal_backend_reply_is_a_generation_error_not_a_successful_answer(self):
        obj={'message':{'content':'partial output'},'done':False}
        row=prediction_record({'id':'fixture'},obj,'ollama',1,'hash')
        self.assertFalse(row['generation_terminal'])
        self.assertEqual(row['generation_error'],'nonterminal_backend_reply')
        self.assertIsNone(row['usage']['completion_tokens'])
        truncated={'choices':[{'message':{'content':'partial'},'finish_reason':'length'}]}
        row=prediction_record({'id':'fixture'},truncated,'openai',1,'hash')
        self.assertTrue(row['truncated'])
        self.assertTrue(row['generation_terminal'])

    def test_generation_error_manifest_survives_resume_without_retry(self):
        from firm_data import write_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);evaluation=root/'eval.jsonl';out=root/'pred.jsonl'
            write_jsonl(evaluation,development_items()[:1])
            argv=['runner','--eval',str(evaluation),'--model','fixture/no-weights',
                  '--model-revision','0'*40,'--out',str(out)]
            response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'partial'},
                                                       'finish_reason':None}]}).encode())
            with patch.object(sys,'argv',argv),patch.object(run_firm_baseline,'repository_revision',return_value='0'*40),\
                    patch.object(run_firm_baseline.urllib.request,'urlopen',return_value=response):
                run_firm_baseline.main()
            with patch.object(sys,'argv',argv),patch.object(run_firm_baseline,'repository_revision',return_value='0'*40),\
                    patch.object(run_firm_baseline.urllib.request,'urlopen') as request:
                run_firm_baseline.main()
                request.assert_not_called()
            meta=json.loads(out.with_suffix('.run.json').read_text())
            self.assertEqual(meta['status'],'complete_with_generation_errors')
            self.assertEqual(meta['generation_error_count'],1)

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
