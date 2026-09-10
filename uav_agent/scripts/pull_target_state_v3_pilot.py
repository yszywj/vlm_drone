#!/usr/bin/env python3
"""PC/WSL read-only-source pilot receiver. Never deletes or extracts anything."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

# On PC copy verify_tar.py alongside this file; no numpy/torch/Isaac needed.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"target_state_v3"))
from verify_tar import verify_tar,file_sha


def atomic_json(path,data):
    fd,name=tempfile.mkstemp(prefix=".pilot-",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(data,f,indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh-target",default="vlm-data")
    p.add_argument("--server-session",required=True)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*",a.ssh_target):
        raise ValueError("unsafe SSH target")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+",a.server_session) or ".." in a.server_session.split("/"):
        raise ValueError("unsafe server session")
    run_id=Path(a.server_session).name
    if not re.fullmatch(r"v3_[a-z0-9_]{1,32}",run_id):
        raise ValueError("invalid V3 session name")
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.output/".receiver.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failures=0
        verified={}
        while True:
            try:
                response=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",a.ssh_target,
                    "cat",a.server_session+"/session.json"],capture_output=True,text=True,check=True,timeout=45)
                state=json.loads(response.stdout)
                contract=state["contract"]
                if contract["run_id"]!=run_id or contract["output"]!=a.server_session or not 0<contract["physical_captures"]<=200:
                    raise ValueError("session identity/count mismatch")
                snapshot=a.output/"session.json"
                if snapshot.exists() and json.loads(snapshot.read_text())["contract"]!=contract:
                    raise ValueError("PC destination belongs to a different pilot contract")
                names=set()
                for entry in state["episodes"]:
                    name=entry["filename"]
                    if not re.fullmatch(re.escape("shard_"+run_id)+r"_[0-9]{6}\.tar",name) or name in names:
                        raise ValueError("unsafe/duplicate archive name")
                    names.add(name)
                    path=a.output/name
                    if not path.exists():
                        subprocess.run(["rsync","-rvh","--no-perms","--no-owner","--no-group",
                            "--partial-dir=.rsync-partial","--timeout=300","--info=progress2",
                            "-e","ssh -o BatchMode=yes -o ConnectTimeout=15","--",
                            a.ssh_target+":"+a.server_session+"/archives/"+name,str(a.output)+"/"],check=True,timeout=900)
                    signature=(entry["sha256"],path.stat().st_size,path.stat().st_mtime_ns)
                    if verified.get(name)!=signature:
                        if path.stat().st_size!=entry["size_bytes"] or file_sha(path)!=entry["sha256"]:
                            raise ValueError(f"PC archive differs: {name}; preserve it and investigate")
                        manifest=verify_tar(path,expected_contract=contract,expected_episode=entry["episode_id"])
                        if manifest["physical_capture_count"]!=20:
                            raise ValueError("partial episode archive")
                        verified[name]=signature
                        print(f"Verified on PC: {name}",flush=True)
                atomic_json(snapshot,state)
                if state.get("last_error"):
                    raise ValueError("Server collector stopped: "+state["last_error"]+"; fix/resume server, then rerun receiver.")
                if state["complete"]:
                    if len(names)*20!=contract["physical_captures"]:
                        raise ValueError("complete session has missing physical captures")
                    atomic_json(a.output/"pc_receipt.json",{"complete":True,"run_id":run_id,
                        "physical_captures":len(names)*20,"archives":{e["filename"]:e["sha256"] for e in state["episodes"]},
                        "archive_and_internal_assets_verified":True,"server_sources_deleted":False})
                    print(f"PC receive complete: {len(names)*20} physical captures; sources retained.",flush=True)
                    return
                failures=0
                print("Waiting for next complete server episode...",flush=True)
            except (subprocess.SubprocessError,OSError) as exc:
                failures+=1
                print(f"Transfer retry {failures}/10: {exc}",file=sys.stderr,flush=True)
                if failures>=10:
                    raise
            time.sleep(5)


if __name__=="__main__":
    run()
