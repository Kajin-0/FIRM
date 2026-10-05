"""Measured smoke gates; CPU fixture results must never be relabeled GPU results."""
import json
import math
import time
from pathlib import Path
import torch
from transformers import TrainerCallback
from firm_data import write_json
from firm_model_profiles import assert_adapter_boundaries


class SmokeMeasurements(TrainerCallback):
    def __init__(self,model,out,modern=False):
        self.model,self.out,self.modern=model,Path(out),modern
        self.records=[];self.tokens=0;self.answer_tokens=0;self.gradient_checks=[]
        self.interruption_requested=False
        self.handle=model.register_forward_pre_hook(self.count_tokens,with_kwargs=True)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()

    def count_tokens(self,module,args,kwargs):
        if self.model.training and 'input_ids' in kwargs:
            mask=kwargs.get('attention_mask')
            self.tokens+=int(mask.sum()) if mask is not None else kwargs['input_ids'].numel()
            labels=kwargs.get('labels')
            if labels is not None:self.answer_tokens+=int((labels!=-100).sum())

    def on_step_begin(self,args,state,control,**kwargs):
        if torch.cuda.is_available():torch.cuda.synchronize()
        self.start=time.perf_counter();self.tokens=0;self.answer_tokens=0

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        names=assert_adapter_boundaries(self.model,self.modern)
        grads={n:p.grad for n,p in self.model.named_parameters() if p.requires_grad}
        if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad):
            raise ValueError('Frozen parameter received a gradient')
        if any(g is None or not torch.isfinite(g).all() for g in grads.values()):
            raise ValueError('An adapter gradient is absent or non-finite')
        nonzero=sum(bool(torch.any(g!=0)) for g in grads.values())
        if nonzero==0:raise ValueError('No adapter receives a nonzero gradient')
        # A gradients may be zero on step 1 because LoRA B initializes to zero.
        self.gradient_checks.append({'step':state.global_step+1,'adapter_tensors':len(names),
                                     'nonzero_gradient_tensors':nonzero,'frozen_gradients':0})

    def on_step_end(self,args,state,control,**kwargs):
        if torch.cuda.is_available():torch.cuda.synchronize()
        elapsed=time.perf_counter()-self.start
        cuda=torch.cuda.is_available()
        self.records.append({'step':state.global_step,'seconds':elapsed,'input_tokens':self.tokens,
            'supervised_tokens':self.answer_tokens,'tokens_per_second':self.tokens/elapsed,
            'allocated_bytes':torch.cuda.memory_allocated() if cuda else None,
            'reserved_bytes':torch.cuda.memory_reserved() if cuda else None,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if cuda else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if cuda else None})
        self.save()
        if self.interruption_requested:
            control.should_save=True;control.should_training_stop=True

    def save(self):
        write_json(self.out/'smoke_measurements.json',{'device':'CUDA' if torch.cuda.is_available() else 'CPU fixture',
            'steps':self.records,'gradient_checks':self.gradient_checks,
            'checkpoint_bytes':sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file() and any(d.startswith('checkpoint-') for d in p.parts)),
            'adapter_bytes':sum(p.stat().st_size for p in self.out.rglob('adapter*') if p.is_file() and not any(d.startswith('checkpoint-') for d in p.parts))})

    def on_train_end(self,args,state,control,**kwargs):self.save()
