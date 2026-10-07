#!/usr/bin/env python3
"""VPS-side bounded checkpoint evacuation and FIRM-only VM cleanup.

Must run on the authorized VPS with a temporary gcloud user login.
Does NOT create VMs or service accounts, and transfers no account credential
to the VM. Every action is project-, zone-, and exact-name-restricted.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

PROJECT="firm-gpu-experiments"
ZONE="us-east4-c"
INSTANCE="firm-4b-v2-2-train-01"
REMOTE="/opt/firmgpu"
RUN="firm4b-v2-2-expanded-v1"
GCLOUD="/home/User/.local/opt/google-cloud-sdk/bin/gcloud"
CONFIG="/home/User/.config/gcloud-firm-temp"
LOCAL=Path("/home/User/FIRM/runs/gpu-results/firm4b-v2-2-expanded-v1")

def log(*parts):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(),*parts,flush=True)

def call(*args,timeout=90,capture=True):
    env=dict(os.environ,CLOUDSDK_CONFIG=CONFIG)
    p=subprocess.run([GCLOUD,*args],env=env,text=True,capture_output=capture,timeout=timeout)
    if p.returncode:
        raise RuntimeError("gcloud "+str(args[:3])+" failed: "+(p.stderr or "")[-650:])
    return p.stdout.strip() if capture else ""

def ssh(command):
    return call("compute","ssh",f"User@{INSTANCE}","--zone",ZONE,"--project",PROJECT,
                "--quiet","--command",command,timeout=75)
def copy_remote(remote,dest,recursive=False):
    args=["compute","scp","--zone",ZONE,"--project",PROJECT,"--quiet"]
    if recursive:args.append("--recurse")
    args.extend([f"User@{INSTANCE}:{remote}",str(dest)])
    call(*args,timeout=240)
def instance_state():
    try:
        return call("compute","instances","describe",INSTANCE,"--zone",ZONE,
                    "--project",PROJECT,"--format=get(status)",timeout=35)
    except Exception:
        return None
def remote_records():
    script="""python3 - <<'PY'
import json
from pathlib import Path
p=Path('/opt/firmgpu/outputs/firm4b-v2-2-expanded-v1')
status=Path('/opt/firmgpu/logs/run_status.json')
complete=[]
if p.exists():
 for c in sorted(p.glob('checkpoint-*')):
  if (c/'trainer_state.json').is_file() and (c/'adapter_model.safetensors').is_file():
   complete.append(c.name)
files=[x.name for x in p.iterdir() if x.is_file()] if p.exists() else []
print(json.dumps({'checkpoints':complete,
 'final':(p/'adapter_model.safetensors').is_file(),
 'files':files,
 'status':status.read_text() if status.exists() else '{}'}))
PY"""
    out=ssh(script)
    return json.loads(out.splitlines()[-1])
def sha_tree(path):
    info={}
    for file in sorted(path.rglob("*")):
        if file.is_file():
            info[str(file.relative_to(path))]={
                "size":file.stat().st_size,
                "sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
    return info
def save_copy(remote,local,recursive=False):
    if local.exists():return
    local.parent.mkdir(parents=True,exist_ok=True)
    tmp=local.with_name(local.name+".partial")
    if tmp.exists():
        if tmp.is_dir():
            import shutil;shutil.rmtree(tmp)
        else:tmp.unlink()
    tmp.mkdir() if recursive else None
    try:
        if recursive:
            copy_remote(remote,tmp,True)
            sub=tmp/Path(remote).name
            if not sub.exists():raise IOError("scp did not download checkpoint folder")
            sub.rename(local)
            tmp.rmdir()
        else:
            copy_remote(remote,tmp,False)
            tmp.rename(local)
        if local.is_dir():
            manifest=sha_tree(local)
            if not any(x.endswith("adapter_model.safetensors") for x in manifest):
                raise IOError("Incomplete model checkpoint copied")
            (local/"vps_copy_integrity.json").write_text(json.dumps(manifest,indent=2)+"\n")
        log("RECOVERED",str(local))
    except Exception:
        log("COPY_FAILED",remote)
        raise
def snapshot():
    r=remote_records()
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/"last_remote_state.json").write_text(json.dumps(r,indent=2)+"\n")
    for name in r["checkpoints"]:
        save_copy(f"{REMOTE}/outputs/{RUN}/{name}",LOCAL/"checkpoints"/name,True)
    for name in ["firm_v22_run.json","smoke_measurements.json","adapter_config.json",
                 "adapter_model.safetensors","tokenizer_config.json","tokenizer.json",
                 "special_tokens_map.json","chat_template.jinja","software-lock.txt"]:
        if name not in r.get("files",[]):
            continue
        try:
            if not r["final"] and name in {"firm_v22_run.json","smoke_measurements.json"}:
                continue
            save_copy(f"{REMOTE}/outputs/{RUN}/{name}",LOCAL/"final"/name)
        except Exception as err:
            log("OPTIONAL_FILE_MISSING",name,str(err)[:80])
    for name in ["startup.log","training.log","run_status.json","software-lock.txt"]:
        try:
            target=LOCAL/"logs"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            # Logs change during training. Copy to temp and replace, not a one-time copy.
            tmp=target.with_name(target.name+".partial")
            copy_remote(f"{REMOTE}/logs/{name}",tmp)
            tmp.rename(target)
        except Exception:pass
    return r

def delete_and_audit():
    state=instance_state()
    if state:
        log("DELETING",INSTANCE,state)
        try:
            call("compute","instances","delete",INSTANCE,"--zone",ZONE,"--project",PROJECT,
                 "--quiet",timeout=210)
        except Exception as ex:
            log("DELETE_RETRY_REQUIRED",str(ex))
    for i in range(12):
        state=instance_state()
        if not state:break
        log("WAITING_FOR_DELETE",state)
        time.sleep(10)
    instances=call("compute","instances","list","--project",PROJECT,
                   "--format=value(name)",timeout=45)
    disks=call("compute","disks","list","--project",PROJECT,
               "--format=value(name)",timeout=45)
    addresses=call("compute","addresses","list","--project",PROJECT,
                    "--format=value(name)",timeout=45)
    reservations=call("compute","reservations","list","--project",PROJECT,
                       "--format=value(name)",timeout=45)
    log("RESOURCE_AUDIT",json.dumps({"instances":instances,"disks":disks,"addresses":addresses,"reservations":reservations}))
    return not any((instances,disks,addresses,reservations))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-minutes",type=int,default=108)
    p.add_argument("--poll-seconds",type=int,default=150)
    a=p.parse_args()
    if not (10<=a.max_minutes<=109 and 30<=a.poll_seconds<=300):raise ValueError("Watch limit invalid")
    LOCAL.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+a.max_minutes*60
    backed=False
    try:
        while time.monotonic()<deadline:
            state=instance_state()
            if not state:
                log("VM_ABSENT")
                break
            log("VM_STATE",state)
            try:
                status=snapshot()
                backed=backed or bool(status["checkpoints"] or status["final"])
                log("CHECKPOINTS",status["checkpoints"],"FINAL",status["final"],
                    "STATE",status["status"])
                s=json.loads(status["status"]).get("status","")
                if s in ("completed_training_pending_evaluation","completed") and status["final"]:
                    final_path=LOCAL/"final"/"adapter_model.safetensors"
                    if final_path.is_file() and final_path.stat().st_size>0:
                        log("TRAINED_FINAL_DOWNLOADED",backed)
                        break
                    log("FINAL_DOWNLOAD_NOT_YET_VERIFIED")
                if s.startswith(("failed_exit","partial_checkpoint_or_failure")):
                    log("TRAINING_FAILED_PARTIAL_SAVED",s)
                    break
            except Exception as ex:
                log("SYNC_RETRY",str(ex)[:350])
            time.sleep(a.poll_seconds)
    finally:
        try:
            clean=delete_and_audit()
        except Exception as ex:
            log("AUDIT_FAILED",str(ex))
            clean=False
        if clean:
            try:
                call("auth","revoke","--all","--quiet",timeout=45)
                log("CREDENTIALS_REVOKED")
            except Exception as ex:log("REVOKE_FAILED",str(ex))
        else:
            log("ACTION_REQUIRED","Resource cleanup not verified; credentials preserved for emergency deletion.")
    log("DONE","checkpoint_available",backed)

if __name__=="__main__":main()
