#!/usr/bin/env python3
"""Generate benchmark answers against an existing localhost compatible model server.

Never loads weights or provisions resources. Uses only prompts, not grading rubrics.
Append-only results resume after interruption; run metadata prevents mixed runs.
"""
import argparse
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from firm_data import digest, require_clean, sha256, write_json
from firm_run_context import repository_revision


def messages_for(item, protocol='v1'):
    if protocol == 'scientific-compact-v2':
        system = ('Solve the infrared detector problem using the supplied assumptions. '
                  'Be precise about equations, units and measurement conventions. '
                  'Keep the explanation within 200 words; state a competing hypothesis or validation check where relevant.')
        if 'grading' in item:
            names = list(item['grading']['quantities'])
            system += (' Return only one JSON object. Put quantities first, then answer. '
                       'quantities maps each required name to {"value": number, "unit": "explicit unit"}; '
                       'answer is your concise explanation. Required names: ' + ', '.join(names))
        return [{'role':'system','content':system},{'role':'user','content':item['prompt']}]
    system = ("You are FIRM, an infrared photodetector science and engineering assistant. "
              "State relevant assumptions, equations and units, calculations, sanity checks, "
              "physical interpretation, competing hypotheses and validation experiments.")
    if "grading" in item:
        # Quantity names only define the response protocol. Never send gold values/units.
        names = list(item["grading"]["quantities"])
        system += (" Return one JSON object with answer (text) and quantities (an object with "
                   "numeric value and explicit unit for each quantity). Required quantity names: " + ", ".join(names))
    return [{"role": "system", "content": system}, {"role": "user", "content": item["prompt"]}]


def parse_answer(raw):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str) and isinstance(parsed.get("quantities"), dict):
            return parsed
    except (ValueError, TypeError):
        pass
    return {"answer": raw, "quantities": {}, "structured_parse_error": True}


def ollama_metadata(base_url, model, revision):
    """Verify an existing local artifact; never pull or invoke a remote cloud model."""
    def get(path, payload=None):
        req = urllib.request.Request(base_url.rstrip('/') + path,
            data=json.dumps(payload).encode() if payload else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    tags = get('/api/tags')['models']
    matched = [m for m in tags if m['name'] == model or m['name'] == model + ':latest']
    if len(matched) != 1 or matched[0].get('remote_host') or matched[0].get('remote_model'):
        raise ValueError('Use one already-installed local Ollama artifact, never a cloud model')
    info = matched[0]
    if info['digest'] != revision:
        raise ValueError('Ollama artifact digest differs from declared revision')
    shown = get('/api/show', {'model': model})
    return {'implementation': 'ollama', 'version': get('/api/version')['version'],
            'artifact_digest': info['digest'], 'upstream_hf_revision': 'unknown',
            'details': shown.get('details'), 'model_info': shown.get('model_info'),
            'renderer_modelfile_sha256': digest(shown.get('modelfile', '')),
            'template_sha256': digest(shown.get('template', '')),
            'default_parameters': shown.get('parameters'), 'cpu': platform.processor(),
            'cpu_count': os.cpu_count(), 'gpu': 'not established; inspect per-response residency'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--backend", choices=['openai', 'ollama'], default='openai')
    ap.add_argument("--thinking", choices=['server-default', 'off', 'on'], default='server-default')
    ap.add_argument("--track", default='E0-standard')
    ap.add_argument('--protocol',choices=['v1','scientific-compact-v2'],default='v1')
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--server-metadata", type=Path, help='Recorded server version, precision, hardware and template configuration')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base = urllib.parse.urlparse(args.base_url)
    if base.hostname not in {"localhost", "127.0.0.1", "::1"} or base.scheme not in {"http", "https"} or base.username or base.password:
        ap.error("This low-cost runner permits only an existing localhost endpoint")
    if args.temperature != 0 or args.max_tokens < 1 or args.context_length < args.max_tokens or args.timeout <= 0:
        ap.error('Primary deterministic protocol requires temperature zero and valid token/context limits')
    items = [r["raw"] for r in require_clean(args.eval, evaluation=True)]
    if len({r["id"] for r in items}) != len(items):
        ap.error("Duplicate eval IDs")
    protocol = [{"id": item["id"], "messages": messages_for(item,args.protocol)} for item in items]
    spec = {"schema_version": "1.0", "model": args.model, "model_revision": args.model_revision,
            "revision_verification": "operator-declared; ensure the server loads this revision",
            "eval_file": str(args.eval), "eval_sha256": sha256(args.eval),
            "protocol_sha256": digest(protocol), "seed": args.seed, "temperature": args.temperature,
            'protocol_name':args.protocol, 'runner_sha256':sha256(Path(__file__)),
            "max_tokens": args.max_tokens, "base_url": args.base_url,
            "backend": args.backend, "thinking": args.thinking, "track": args.track,
            "context_length": args.context_length, "sampling": {'presence_penalty': 0, 'top_p': 1},
            "server_metadata": json.loads(args.server_metadata.read_text()) if args.server_metadata else None,
            "git_sha": repository_revision()}
    if args.dry_run:
        print(json.dumps({**spec, "items": len(items), "status": "dry_run_no_inference"}, indent=2))
        return
    if args.backend == 'ollama':
        if args.thinking == 'server-default':
            ap.error('Declare --thinking off/on for reproducible Ollama comparisons')
        spec['server_metadata'] = ollama_metadata(args.base_url, args.model, args.model_revision)
        spec['revision_verification']='local artifact digest verified; upstream HF revision unknown'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.out.with_suffix(".run.json")
    done = set()
    if args.out.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text())["spec"] != spec:
            ap.error("Resume manifest differs; use a new output path")
        for line in args.out.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in done or row["id"] not in {r["id"] for r in items}:
                ap.error("Invalid existing prediction IDs")
            done.add(row["id"])
    else:
        if meta_path.exists():
            ap.error("Run manifest exists without predictions; use a new path")
        write_json(meta_path, {"spec": spec, "timestamp": datetime.now(timezone.utc).isoformat(), "status": "started"})
    for item in items:
        if item["id"] in done:
            continue
        payload = {"model": args.model, "messages": messages_for(item,args.protocol), "temperature": args.temperature,
                   "seed": args.seed, "max_tokens": args.max_tokens, 'presence_penalty': 0, 'top_p': 1}
        route = '/chat/completions'
        if args.backend == 'ollama':
            payload = {'model': args.model, 'messages': messages_for(item,args.protocol), 'stream': False,
                       'think': args.thinking == 'on', 'options': {'temperature': 0, 'seed': args.seed,
                       'num_predict': args.max_tokens, 'num_ctx': args.context_length,
                       'presence_penalty': 0, 'top_p': 1, 'top_k': 0}}
            route = '/api/chat'
        elif args.thinking != 'server-default':
            payload['chat_template_kwargs'] = {'enable_thinking': args.thinking == 'on'}
        started = time.perf_counter()
        req = urllib.request.Request(args.base_url.rstrip("/") + route,
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            obj = json.load(response)
        elapsed = time.perf_counter() - started
        if args.backend == 'ollama':
            choice = {'message': obj['message'], 'finish_reason': obj.get('done_reason')}
            choice['message']['reasoning_content'] = obj['message'].get('thinking')
            obj['usage'] = {'prompt_tokens': obj.get('prompt_eval_count'), 'completion_tokens': obj.get('eval_count')}
        else:
            choice = obj["choices"][0]
        raw = choice["message"].get("content")
        if not isinstance(raw, str):
            raise ValueError("Server returned no text content")
        row = {"id": item["id"], **parse_answer(raw), "raw_response": raw,
               "finish_reason": choice.get("finish_reason"), "usage": obj.get("usage"),
               "reasoning_content": choice["message"].get("reasoning_content"),
               "latency_seconds": elapsed, "raw_server_response": obj,
               "eval_sha256": spec["eval_sha256"]}
        with args.out.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(item["id"], choice.get("finish_reason"), flush=True)
    meta = json.loads(meta_path.read_text())
    meta.update({"status": "complete", "predictions_sha256": sha256(args.out)})
    write_json(meta_path, meta)


if __name__ == "__main__":
    main()
