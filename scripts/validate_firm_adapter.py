"""Reload an actual saved smoke adapter and compare its recorded probe logits."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from firm_data import sha256,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    import torch
    from peft import PeftModel
    from firm_model_profiles import load_model
    run=json.loads((args.run/'experiment_manifest.json').read_text())
    ref=json.loads((args.run/'reload_reference.json').read_text())
    if not torch.cuda.is_available():raise ValueError('Actual adapter validation requires the intended CUDA environment')
    model_args=SimpleNamespace(model=run['base_model'],model_revision=run['base_model_revision'],model_profile=run['model_profile'])
    dtype=torch.bfloat16 if run['hardware']['precision']=='bf16' else torch.float16
    base,_,_=load_model(model_args,dtype)
    model=PeftModel.from_pretrained(base,str(args.run));model.eval()
    batch={k:torch.tensor(ref[k],device='cuda') for k in ('input_ids','attention_mask')}
    with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
        logits=model(**batch,use_cache=False).logits[0,-1].float().cpu()
    expected=torch.tensor(ref['last_logits']);actual=logits[ref['logit_indices']]
    passed=torch.allclose(expected,actual,atol=ref['atol'],rtol=ref['rtol']) and int(logits.argmax())==ref['greedy_next_token']
    write_json(args.run/'adapter_reload_validation.json',{'passed':passed,'gpu':torch.cuda.get_device_name(0),
        'max_sampled_logit_absolute_error':float((expected-actual).abs().max()),
        'greedy_next_token':int(logits.argmax()),'reference_sha256':sha256(args.run/'reload_reference.json'),
        'adapter_bytes':sum(p.stat().st_size for p in args.run.glob('adapter*') if p.is_file()),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    if not passed:raise ValueError('Adapter reload gate failed')
    print('PASS actual adapter reload')


if __name__=='__main__':main()
