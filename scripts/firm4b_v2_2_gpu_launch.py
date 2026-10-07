#!/usr/bin/env python3
"""Launch exactly one two-hour FIRM 4B v2.2 VM, then start a VPS checkpoint watchdog.

The user's short-lived Google login must already exist in dedicated CLOUDSDK_CONFIG.
The VM receives no Google credentials or service account. Rendering the startup
script pins the exact committed Git SHA. This is research training only.
"""
from __future__ import annotations
import argparse,os,subprocess,tempfile,json,hashlib
from pathlib import Path
from firm4b_v2_2_vps_watchdog import GCLOUD,CONFIG,PROJECT,ZONE,INSTANCE

ROOT=Path(__file__).resolve().parents[1]
STARTUP=ROOT/"scripts/firm4b_v2_2_gpu_startup.sh"
LOG_ROOT=ROOT/"runs/gpu-results/firm4b-v2-2-expanded-v1"
WATCH="firm4b-v2-2-evacuate"

def cmd(args,timeout=120):
    p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("Command failed: "+str(args[:4])+"\n"+p.stderr[-1200:])
    return p.stdout.strip()

def gcloud(*args,timeout=120):
    env=dict(os.environ,CLOUDSDK_CONFIG=CONFIG)
    p=subprocess.run([GCLOUD,*args],env=env,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError("gcloud "+str(args[:3])+" failed: "+p.stderr[-1200:])
    return p.stdout.strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",action="store_true",help="Create a billable Google GPU (requires prior user authorization).")
    a=ap.parse_args()
    source_sha=cmd(["git","rev-parse","HEAD"])
    if len(source_sha)!=40:raise ValueError("Unexpected repository revision")
    if a.start and cmd(["git","status","--porcelain"]):
        raise ValueError("Uncommitted changes: commit/test before provisioning cloud resources")
    template=STARTUP.read_text()
    if template.count("@FIRM_PINNED_COMMIT@")!=2:
        raise ValueError("Incorrect commit placeholder count")
    rendered=template.replace("@FIRM_PINNED_COMMIT@",source_sha)
    if not rendered.startswith("#!/usr/bin/env bash"):raise ValueError("Unexpected startup script")
    with tempfile.TemporaryDirectory(prefix="firm4b-render-") as t:
        path=Path(t)/"startup.sh";path.write_text(rendered);path.chmod(0o600)
        cmd(["bash","-n",str(path)])
        print(json.dumps({"project":PROJECT,"zone":ZONE,"vm":INSTANCE,"machine":"g2-standard-8",
                          "max_run_duration_seconds":7200,"termination_action":"DELETE",
                          "service_account":"none","source_git_sha":source_sha,
                          "rendered_startup_sha256":hashlib.sha256(rendered.encode()).hexdigest(),
                          "stage":"authorization_required" if not a.start else "launching",
                          "research_only":True},indent=2),flush=True)
        if not a.start:return
        if "@" in source_sha:raise ValueError("Invalid SHA")
        identity=gcloud("auth","list","--filter=status:ACTIVE","--format=value(account)")
        if not identity:raise PermissionError("No active user login; complete secure gcloud browser login")
        billing=gcloud("billing","projects","describe",PROJECT,"--format=json(billingEnabled)")
        if json.loads(billing).get("billingEnabled") is not True:
            raise RuntimeError("FIRM project has no active billing account")
        for what in ("instances","disks","addresses","reservations"):
            resources=gcloud("compute",what,"list","--project",PROJECT,"--format=value(name)")
            if resources:raise RuntimeError("Unexpected existing cloud resource "+what+": "+resources)
        print("Clean resource preflight confirmed; provisioning 1 L4 for at most 7200 seconds.",flush=True)
        request=[
            "compute","instances","create",INSTANCE,
            "--project",PROJECT,"--zone",ZONE,"--machine-type=g2-standard-8",
            "--image-family=common-cu129-ubuntu-2204-nvidia-580",
            "--image-project=deeplearning-platform-release",
            "--boot-disk-size=100GB","--boot-disk-type=pd-balanced",
            "--no-service-account","--no-scopes",
            "--maintenance-policy=TERMINATE","--no-restart-on-failure",
            "--max-run-duration=2h","--instance-termination-action=DELETE",
            "--metadata-from-file=startup-script="+str(path),
            "--quiet","--format=json(name,status,scheduling.maxRunDuration,scheduling.instanceTerminationAction)"
        ]
        result=gcloud(*request,timeout=180)
        print("CREATE_RESULT",result,flush=True)
    details=json.loads(gcloud("compute","instances","describe",INSTANCE,
                               "--project",PROJECT,"--zone",ZONE,
                               "--format=json(status,scheduling.maxRunDuration,scheduling.instanceTerminationAction,serviceAccounts,disks)"))
    sched=details.get("scheduling",{})
    seconds=int(sched.get("maxRunDuration",{}).get("seconds",0))
    if sched.get("instanceTerminationAction")!="DELETE" or seconds!=7200:
        raise RuntimeError("Provider auto-delete guard not verified. Destroy VM NOW.")
    if details.get("serviceAccounts"):
        raise RuntimeError("Unexpected VM service account. Destroy VM NOW.")
    if not all(d.get("autoDelete") for d in details.get("disks",[])):
        raise RuntimeError("Some disk does not have autoDelete=true. Destroy VM NOW.")
    LOG_ROOT.mkdir(parents=True,exist_ok=True)
    running=subprocess.run(["tmux","has-session","-t",WATCH],capture_output=True).returncode==0
    if running:raise RuntimeError("Checkpoint watchdog tmux name is occupied")
    command=(f"cd {ROOT} && python3 scripts/firm4b_v2_2_vps_watchdog.py "
             f"--max-minutes 108 --poll-seconds 150 >> "
             f"{LOG_ROOT}/watchdog.log 2>&1")
    cmd(["tmux","new-session","-d","-s",WATCH,command])
    print("GUARDS_VERIFIED",seconds,"sec provider auto-delete, no VM service account, disk autoDelete.")
    print("WATCHDOG_STARTED",WATCH,"log",LOG_ROOT/"watchdog.log")
    print("Training is started automatically by instance metadata; watch logs/checkpoints on VPS.")

if __name__=="__main__":
    main()
