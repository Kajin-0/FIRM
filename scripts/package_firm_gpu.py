"""Build a hash-verified, credential-free run package or collect explicit run artifacts.

No model weights, environment directories, Git internals, credentials or .env
files enter the execution package. Runtime collection has a strict filename
allowlist and rejects symlinks. Existing archives are never overwritten.
"""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path
from firm_data import sha256,write_json


def clean_checkout():
    # Ignore untracked local outputs; require committed tracked code/data.
    if subprocess.check_output(['git','diff','HEAD','--name-only'],text=True).strip():
        raise ValueError('Commit tracked changes before packaging')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def package_files():
    tracked=subprocess.check_output(['git','ls-files','-z'],text=True).split('\0')
    chosen=[]
    for name in tracked:
        if not name:continue
        path=Path(name)
        if (name.startswith(('scripts/','configs/','tests/','docs/FIRM3_','data/reviews/','data/manifests/','data/processed/firm3_reviewed_seed_v3/','data/processed/firm3_synthetic_quant_v1/'))
            or name in {'data/processed/firm_rewritten_large_sft.jsonl','data/audits/firm3_2026-10-05/dataset_audit.json'}
            or name.startswith('data/processed/firm_v2') and 'expert' in name and path.suffix=='.jsonl'
            or (name.startswith('evals/') and len(path.parts)==2 and path.suffix in {'.json','.jsonl'})
            or name.startswith('requirements') or name in {'README.md','.gitignore'}):
            if path.is_symlink() or not path.is_file():raise ValueError('Invalid package file: '+name)
            if any(part.startswith('.env') or part in {'.git','.ssh'} for part in path.parts) or path.suffix in {'.key','.pem'}:
                raise ValueError('Credential path forbidden')
            chosen.append(path)
    return sorted(chosen)


def artifact_files(run):
    root_names={'experiment_manifest.json','reload_reference.json','adapter_reload_validation.json',
        'smoke_measurements.json','software-lock.txt','trainer_state.json','training_args.bin',
        'adapter_config.json','adapter_model.safetensors','adapter_model.bin','tokenizer.json',
        'tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt'}
    checkpoint_names={'adapter_config.json','adapter_model.safetensors','adapter_model.bin','optimizer.pt',
        'scheduler.pt','rng_state.pth','scaler.pt','trainer_state.json','training_args.bin',
        'tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt'}
    chosen=[]
    for path in sorted(run.rglob('*')):
        if path.is_symlink():raise ValueError('Artifact symlink forbidden')
        if not path.is_file():continue
        relative=path.relative_to(run)
        if (len(relative.parts)==1 and path.name in root_names or
            len(relative.parts)==2 and relative.parts[0].startswith('checkpoint-') and path.name in checkpoint_names):
            chosen.append(path)
    if not any(p.name=='experiment_manifest.json' for p in chosen):raise ValueError('Missing experiment manifest')
    return chosen


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--collect',type=Path);args=ap.parse_args()
    if args.out.exists():ap.error('Archive exists; choose a new path')
    files=artifact_files(args.collect) if args.collect else package_files()
    root=args.collect.resolve() if args.collect else Path('.').resolve()
    manifest={'schema_version':'1.0','kind':'measured_run_artifacts' if args.collect else 'FIRM_GPU_execution',
        'git_sha':None if args.collect else clean_checkout(),
        'assets':[{'path':str(p.resolve().relative_to(root)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in files],
        'reminder':'Provisioning and GPU spending require explicit authorization; collect artifacts and stop the machine.',
        'commands':['bash scripts/bootstrap_firm_gpu.sh E1B','FIRM_RUN_DIR=/persistent/FIRM/E1B bash scripts/firm_gpu_run.sh E1B preflight',
                    'FIRM_RUN_DIR=/persistent/FIRM/E1B bash scripts/firm_gpu_run.sh E1B train',
                    'FIRM_RUN_DIR=/persistent/FIRM/E1B bash scripts/firm_gpu_run.sh E1B resume /persistent/FIRM/E1B/checkpoint-1',
                    'FIRM_RUN_DIR=/persistent/FIRM/E1B bash scripts/firm_gpu_run.sh E1B reload',
                    'FIRM_RUN_DIR=/persistent/FIRM/E1B bash scripts/firm_gpu_run.sh E1B collect']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.out,'w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.resolve().relative_to(root)),recursive=False)
        content=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
        member=tarfile.TarInfo('firm_gpu_artifact_manifest.json' if args.collect else 'firm_gpu_package_manifest.json')
        member.size=len(content);archive.addfile(member,io.BytesIO(content))
    write_json(args.out.with_suffix(args.out.suffix+'.manifest.json'),{**manifest,'archive_sha256':sha256(args.out)})
    print(args.out,sha256(args.out),len(files),'files')


if __name__=='__main__':main()
