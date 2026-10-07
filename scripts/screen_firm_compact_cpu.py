#!/usr/bin/env python3
"""Bounded CPU-only development screening, not final quantitative E0.

Use SAME instructions, decoding and time budget across candidates, report raw
responses and latency. No benchmark secrets leave the local host.
"""
from __future__ import annotations
import argparse,json,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path

PROMPTS=[
("materials_czt","domain","What is CdZnTe and why is it used for mercury cadmium telluride epitaxy?"),
("materials_insb","domain","Explain InSb's chemical composition and approximate room-temperature bandgap."),
("bandgap_conversion","domain","A HgCdTe photodetector has a cutoff wavelength of 8.1 micrometers at 300 K. Explain how one would estimate x, the cadmium mole fraction, distinguishing photons from bandgap energy."),
("substrates","domain","Which crystalline substrates are standard and alternative choices for HgCdTe epitaxial films? Highlight lattice mismatch."),
("detectivity","domain","Given a 0.25 mm^2 detector and a noise equivalent power spectral density of 2e-11 W/sqrt(Hz), calculate D* in Jones and show the area conversion."),
("noise","domain","Explain why white Johnson voltage ASD and integrated RMS noise are not interchangeable in a detector measurement."),
("scope_food","outside","Write a detailed chocolate chip cookie recipe with ingredients."),
("scope_history","outside","Tell me the detailed career of the actor who starred in a 1990s romantic comedy."),
("scope_border","domain","Derive the minority carrier diffusion length in a semiconductor and state the assumptions."),
("identity","metadata","What underlying model are you, and are your weights specialized for infrared physics?")
]
COMMON_SYSTEM=("You are being evaluated as a candidate for a compact infrared photonics "
"and semiconductor science specialist named FIRM. Answer in-domain science directly "
"and precisely. Refuse wholly unrelated questions in one sentence. "
"Questions about your underlying model name and weights must be answered accurately. "
"No invented equations, measurements, citations or semiconductor materials. "
"Use equations and units where relevant. Do not reveal chain-of-thought.")
def run(model,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    done={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);done[r["id"]]=r
    for id_,category,prompt in PROMPTS:
        if id_ in done:continue
        data={"model":model,"messages":[{"role":"system","content":COMMON_SYSTEM},{"role":"user","content":prompt}],
              "stream":False,"think":False,"keep_alive":"5m",
              "options":{"temperature":0,"num_ctx":2048,"num_predict":230}}
        req=urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                    data=json.dumps(data).encode(),headers={"Content-Type":"application/json"})
        t=time.monotonic()
        try:
            with urllib.request.urlopen(req,timeout=150) as res: obj=json.load(res)
            msg=obj.get("message",{})
            response=msg.get("content","")
            err=None
            tokens=obj.get("eval_count")
            thinking_tokens=len(msg.get("thinking","") or "")
            duration=obj.get("eval_duration",0)/1e9
        except Exception as exc:
            response="";err=type(exc).__name__+": "+str(exc);tokens=None;thinking_tokens=None;duration=None
        record={"id":id_,"category":category,"model":model,"prompt":prompt,"response":response,
                "error":err,"tokens":tokens,"thinking_chars":thinking_tokens,
                "elapsed_seconds":round(time.monotonic()-t,2),"decode_seconds":duration,
                "time_utc":datetime.now(timezone.utc).isoformat()}
        with path.open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
        print(json.dumps({k:record[k] for k in ("id","model","tokens","thinking_chars","elapsed_seconds","error")}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.model,a.output)
