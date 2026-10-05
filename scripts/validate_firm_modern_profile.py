"""Exercise native hybrid/VLM class + PEFT + TRL using tiny random CPU weights.

This is an inexpensive software test, not Qwen3.5-9B training. No pretrained
weights are downloaded. GPU/BF16/full-size/NF4 compatibility remains unmeasured.
"""
import argparse
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path
from firm_data import sha256,write_json


def tiny_config():
    from transformers import Qwen3_5Config
    return Qwen3_5Config(text_config={'model_type':'qwen3_5_text','vocab_size':64,'hidden_size':32,
        'intermediate_size':64,'num_hidden_layers':4,'num_attention_heads':4,'num_key_value_heads':2,
        'head_dim':8,'linear_key_head_dim':8,'linear_value_head_dim':8,
        'linear_num_key_heads':2,'linear_num_value_heads':4,'linear_conv_kernel_dim':4,
        'layer_types':['linear_attention']*3+['full_attention'],'max_position_embeddings':128,
        'rope_parameters':{'rope_type':'default','rope_theta':10000,'partial_rotary_factor':1,
                           'mrope_section':[1,1,2],'mrope_interleaved':True},
        'pad_token_id':0,'eos_token_id':1,'use_cache':False},
        vision_config={'depth':1,'hidden_size':16,'intermediate_size':32,'num_heads':2,
        'out_hidden_size':32,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':1,
        'num_position_embeddings':16},image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)


def run_fixture(out,tokenizer_files):
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer,PreTrainedTokenizerFast,Qwen3_5ForConditionalGeneration,TrainerCallback
    from peft import LoraConfig,get_peft_model,PeftModel
    from trl import SFTConfig,SFTTrainer
    from firm_model_profiles import modern_targets,completion_encoding,completion_collator,assert_adapter_boundaries
    from firm_training_measurements import SmokeMeasurements
    torch.set_num_threads(1);torch.manual_seed(42)
    native=Qwen3_5ForConditionalGeneration(tiny_config())
    targets=modern_targets(dict(native.named_modules()))
    if len(targets)!=12:raise ValueError('Tiny fixture expected 4*3 language MLP targets')
    model=get_peft_model(native,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=targets))
    assert_adapter_boundaries(model,True)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({str(i):i for i in range(64)},unk_token='0')),
        unk_token='0',pad_token='0',eos_token='1')
    tokenized={'input_ids':list(range(2,18)),'attention_mask':[1]*16,'labels':[-100]*5+list(range(7,18))}
    dataset=Dataset.from_list([tokenized,tokenized])
    class StopAfterOne(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step==1:control.should_training_stop=True;control.should_save=True
    settings=SFTConfig(output_dir=str(out/'checkpoints'),max_steps=2,per_device_train_batch_size=1,
        gradient_accumulation_steps=1,learning_rate=1e-3,save_steps=1,save_total_limit=2,
        max_length=32,completion_only_loss=True,dataset_kwargs={'skip_prepare_dataset':True},
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        packing=False,remove_unused_columns=False,report_to='none',seed=42,data_seed=42,logging_steps=1,optim='adamw_torch',use_cpu=True,
        bf16=False,fp16=False)
    measure=SmokeMeasurements(model,out,True)
    trainer=SFTTrainer(model=model,args=settings,processing_class=tokenizer,data_collator=completion_collator(tokenizer),
        train_dataset=dataset,callbacks=[StopAfterOne(),measure])
    trainer.train()
    if trainer.state.global_step!=1:raise ValueError('Fixture interruption did not stop at step 1')
    trainer.remove_callback(StopAfterOne)
    trainer.train(resume_from_checkpoint=str(out/'checkpoints/checkpoint-1'))
    if trainer.state.global_step!=2:raise ValueError('Optimizer/RNG resume did not reach step 2')
    if any(not torch.equal(p.detach(),frozen[n]) for n,p in model.named_parameters() if not p.requires_grad):
        raise ValueError('A frozen tiny parameter changed')
    model.eval();probe=completion_collator(tokenizer)([tokenized]);probe.pop('labels')
    with torch.no_grad():expected=model(**probe,use_cache=False).logits
    model.save_pretrained(out/'adapter')
    measure.save()
    torch.manual_seed(42)
    reload_base=Qwen3_5ForConditionalGeneration(tiny_config())
    reloaded=PeftModel.from_pretrained(reload_base,str(out/'adapter'));reloaded.eval()
    with torch.no_grad():actual=reloaded(**probe,use_cache=False).logits
    if not torch.allclose(expected,actual,atol=1e-6,rtol=1e-6):
        old=dict(model.named_parameters());new=dict(reloaded.named_parameters())
        differences={n:float((p.detach()-new[n].detach()).abs().max()) for n,p in old.items() if n in new}
        new_buffers=dict(reloaded.named_buffers())
        buffers={n:float((p.detach()-new_buffers[n].detach()).abs().max()) for n,p in model.named_buffers() if n in new_buffers}
        write_json(out/'adapter_reload_failure.json',{'max_logit_error':float((expected-actual).abs().max()),
            'parameter_differences':{n:v for n,v in differences.items() if v!=0},
            'buffer_differences':{n:v for n,v in buffers.items() if v!=0},
            'attention_implementations':[model.config.get_text_config()._attn_implementation,reloaded.config.get_text_config()._attn_implementation],
            'mode_differences':{n:[m.training,dict(reloaded.named_modules())[n].training] for n,m in model.named_modules() if n in dict(reloaded.named_modules()) and m.training!=dict(reloaded.named_modules())[n].training},
            'old_config':model.config.to_dict(),'new_config':reloaded.config.to_dict()})
        raise ValueError('CPU adapter reload changed logits; diagnostic saved')
    mask_checks=[]
    if tokenizer_files:
        real_tokenizer=AutoTokenizer.from_pretrained(tokenizer_files,local_files_only=True)
        for row in [json.loads(s) for s in Path('data/processed/firm3_reviewed_seed_v3/train.jsonl').read_text().splitlines()][:8]:
            encoded=completion_encoding(real_tokenizer,row['messages'],512)
            mask_checks.append({'id':row['id'],'tokens':len(encoded['input_ids']),
                                'supervised_tokens':sum(t!=-100 for t in encoded['labels'])})
    result={'status':'PASS tiny CPU software fixture only','python':platform.python_version(),
        'software_versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','trl','datasets','accelerate','bitsandbytes']},
        'native_class':'Qwen3_5ForConditionalGeneration','hybrid_layers':['linear_attention']*3+['full_attention'],
        'target_modules':targets,'vision_and_all_frozen_tensors_unchanged':True,'completion_masks':mask_checks,
        'gradient_checks':measure.gradient_checks,'checkpoint_resume_success':True,'adapter_reload_success':True,
        'reload_max_absolute_error':float((expected-actual).abs().max()),
        'fixture_parameters':sum(p.numel() for p in model.parameters()),'GPU_measured':False,
        'limitations':['Random tiny architecture, not pretrained Qwen3.5-9B','CPU float32, not GPU BF16 or 4-bit',
                      'Real pinned tokenizer masks tested separately; vision inference not tested'],
        'native_source_sha256':sha256(Path(__import__('inspect').getfile(Qwen3_5ForConditionalGeneration)))}
    write_json(out/'profile_validation.json',result);return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tokenizer-files',type=Path);args=ap.parse_args()
    if args.out.exists():ap.error('Existing validation artifacts; choose a new directory')
    args.out.mkdir(parents=True)
    result=run_fixture(args.out,args.tokenizer_files)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
