"""Explicit loaders/adapter boundaries; GPU libraries are imported only on demand."""
import math
import re
import importlib.metadata

QWEN35_REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
QWEN35_4B_REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
LANGUAGE_MLP=re.compile(r'^model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$')


def validate_runtime_versions(profile):
    expected={'torch':'2.9.1'}
    expected.update({'transformers':'5.18.0','peft':'0.21.2','trl':'1.14.1','accelerate':'1.15.0','datasets':'4.7.0','bitsandbytes':'0.50.2'}
        if profile in ('qwen35_text_bf16','qwen35_4b_text_bf16') else
        {'transformers':'4.57.6','peft':'0.18.1','trl':'0.24.0','accelerate':'1.11.0','datasets':'4.4.1','bitsandbytes':'0.48.2'})
    for package,version in expected.items():
        actual=importlib.metadata.version(package)
        if actual.split('+')[0]!=version:raise ValueError('Profile version mismatch: '+package+' '+actual+' != '+version)
    return {p:importlib.metadata.version(p) for p in expected}


def validate_profile(args):
    if args.model_profile=='qwen35_text_bf16':
        if args.model!='Qwen/Qwen3.5-9B' or args.model_revision!=QWEN35_REVISION:
            raise ValueError('Modern smoke profile pins the actual Qwen3.5-9B candidate/revision')
        if args.quantization!='none' or args.lora_dropout!=0:
            raise ValueError('First modern gate uses BF16 LoRA and zero adapter dropout; NF4 is not yet validated')
        if args.max_steps not in {1,2,3} or not args.max_train_examples or not 8<=args.max_train_examples<=16:
            raise ValueError('Unmeasured modern profile is limited to 8..16 reviewed rows and 1..3 steps; E2 is gated')
    elif args.model_profile=='qwen35_4b_text_bf16':
        if args.model!='Qwen/Qwen3.5-4B' or args.model_revision!=QWEN35_4B_REVISION:
            raise ValueError('Compact specialist profile requires exact pinned Qwen3.5-4B revision')
        if args.quantization!='none' or args.lora_dropout!=0:
            raise ValueError('Compact Qwen3.5 BF16 LoRA profile forbids unvalidated NF4/dropout')
    elif args.model_profile=='legacy_causal':
        if args.model.startswith(('Qwen/Qwen3.5','Qwen/Qwen3.6','Qwen/Qwen3.8','google/gemma-4','mistralai/Ministral-3')):
            raise ValueError('Do not route modern vision/hybrid checkpoints through the legacy causal profile')
        if args.quantization!='nf4':
            raise ValueError('Legacy smoke profile specifies NF4')
    else:
        raise ValueError('Unknown model profile')


def modern_targets(names):
    return sorted(name for name in names if LANGUAGE_MLP.fullmatch(name))


def load_model(args, compute_dtype):
    import torch
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.model_revision,trust_remote_code=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    if args.model_profile in ('qwen35_text_bf16','qwen35_4b_text_bf16'):
        from transformers import Qwen3_5ForConditionalGeneration
        if compute_dtype!=torch.bfloat16:
            raise ValueError('Modern profile requires GPU BF16 support')
        model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,revision=args.model_revision,
            trust_remote_code=False,dtype=compute_dtype,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
        # The pinned release also stores optional MTP tensors. The native class
        # does not implement MTP; no other missing/mismatched keys are acceptable.
        unexpected=loading.get('unexpected_keys',[])
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs') or any(not k.startswith('mtp.') for k in unexpected):
            raise ValueError('Native checkpoint loading boundaries changed: '+str(loading))
        model.firm_loading_info=loading
        targets=modern_targets(dict(model.named_modules()))
        if len(targets)!=96:
            raise ValueError('Expected 96 language MLP targets; module boundaries changed')
        model.requires_grad_(False)
    else:
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM,BitsAndBytesConfig
        model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.model_revision,
            trust_remote_code=False,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=compute_dtype),
            device_map={'':0},torch_dtype=compute_dtype)
        model=prepare_model_for_kbit_training(model)
        targets='all-linear'
    model.config.use_cache=False
    if hasattr(model.config,'text_config'):model.config.text_config.use_cache=False
    return model,tokenizer,targets


def completion_encoding(tokenizer,messages,max_length):
    """Exact template prefix boundary; no guessing or truncation of supervised text."""
    if messages[-1]['role']!='assistant' or sum(m['role']=='assistant' for m in messages)!=1:
        raise ValueError('One final assistant completion required')
    if any(any(t in m['content'] for t in ('<|image_pad|>','<|video_pad|>','<|vision_start|>')) for m in messages):
        raise ValueError('Text-only profile forbids visual placeholders')
    prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(prefix):raise ValueError('Chat template completion boundary changed')
    ids=tokenizer.encode(full,add_special_tokens=False);prompt_ids=tokenizer.encode(prefix,add_special_tokens=False)
    if ids[:len(prompt_ids)]!=prompt_ids:raise ValueError('Tokenizer merges across the prompt boundary')
    if len(ids)>max_length or len(prompt_ids)>=len(ids):raise ValueError('Overlength or empty assistant target')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prompt_ids)+ids[len(prompt_ids):]}


def completion_collator(tokenizer):
    def collate(examples):
        import torch
        size=max(len(r['input_ids']) for r in examples)
        return {key:torch.tensor([r[key]+[pad]*(size-len(r[key])) for r in examples])
                for key,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}
    return collate


def assert_adapter_boundaries(model,modern=False):
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name in trainable):
        raise ValueError('Only LoRA parameters may train')
    if modern and any('model.language_model.layers.' not in n or '.mlp.' not in n for n in trainable):
        raise ValueError('Vision/attention/DeltaNet or other frozen module became trainable')
    return trainable


def frozen_fingerprint(model):
    """Sample every frozen tensor after loading; full tiny-model equality is tested separately."""
    import hashlib
    out=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            out.update(name.encode());out.update(p.detach().reshape(-1)[:16].float().cpu().numpy().tobytes())
    return out.hexdigest()
