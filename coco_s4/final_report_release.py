"""Register deterministic final reporting on the existing serial controller.

This CPU-only derived-report stage consumes accepted receipts and fixed official
metrics; it does not create another scientific training/evaluation candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT
from .runtime import DATA_FILES, DATA_ROOT, ROOT, load, now, sha, write_once

EX = ROOT / "execution"
PYTHON = "/project/guoqinglong/miniconda3/envs/cfbdm/bin/python"
STAGES = ("teacher200", "teacher_eval", "calibration", "fourier24", "export25k",
          "clean200", "s4_200", "official_eval")

READ_DEPENDENCY = r'''
from pathlib import Path
import hashlib,json,os,sys
o=Path(sys.argv[1]);run=sys.argv[2];source=sys.argv[3]
s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==run and s['source_commit']==source
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if s.get('state')!='WAITING_PHASE_RELEASE' or s.get('current_stage')!='final_report' or s.get('active_job') is not None or s.get('active_child') is not None:
 print(json.dumps(dict(state='WAITING_SERIAL_SLOT',native_state=s.get('state'),current_stage=s.get('current_stage'),active_job=s.get('active_job'))));sys.exit(0)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
deps={}
for name in ('teacher200','teacher_eval','calibration','fourier24','export25k','clean200','s4_200','official_eval'):
 stage=s.get('stages',{}).get(name,{})
 path=o/name/'accepted.json';r=json.loads(path.read_text());digest=sha(path)
 assert stage.get('state')=='ACCEPTED' and stage.get('receipt_sha256')==digest and r.get('state')=='ACCEPTED'
 deps[name]=dict(accepted_path=str(path),accepted_sha256=digest)
ev=json.loads((o/'official_eval/accepted.json').read_text())
assert ev.get('image_count')==5000 and ev.get('classes')==80 and ev.get('paired_metrics')
sys.path.insert(0,sys.argv[4]);from coco_s4.serial import capacity
cap=capacity(o)
print(json.dumps(dict(state='READY_FOR_FINAL_REPORT' if cap.get('admitted') else 'WAITING_RESOURCES',
 phase='final_report',dependencies=deps,capacity=cap,controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def source_tag(tag):
    if not re.fullmatch(r"coco_official_eval_[a-z0-9_]{1,36}", tag):
        raise ValueError("Pass the accepted official-evaluation validation tag")
    return tag


def current_source():
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError("Final-report source must be committed and clean")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(ROOT), "ls-remote", "origin",
        "refs/heads/codex/s4-coco-risk-fallback-20261027"], text=True).split()
    if not remote or remote[0] != head:
        raise ValueError("Final-report candidate must be published and remotely verified")
    return head


def read_dependency(transport):
    active = OWNED / "repos" / WORKFLOW
    r = transport.run([PYTHON, "-B", "-c", READ_DEPENDENCY, WORKFLOW_OUTPUT.as_posix(), WORKFLOW,
                       load(EX/(WORKFLOW+".json"))["source"]["commit"], active.as_posix()])
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace")[-2500:])
    return json.loads(r.stdout)


def build(tag, head, observation):
    if observation.get("state") != "READY_FOR_FINAL_REPORT":
        raise ValueError("Existing serial chain/resources do not permit final-report preparation")
    code = load(EX / (tag + ".json"))
    if code.get("source", {}).get("commit") != head:
        raise ValueError("Accepted official-evaluation release does not use current report code")
    names = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
    selected = [n for n in names if n.startswith(("coco_s4/", "ue_framework/", "configs/")) or n in
                ("docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json",
                 "docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml")]
    actual = load(EX / "ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json")
    release = dict(schema="s4.coco.release.v1", phase="final_report", workflow_run_id=WORKFLOW,
        source_commit=head, scientific_parent="b1e29510e7b1ed5fa0fe361a27132d8790e65c14",
        victim_parent="ab626917f20917e58cf30033ac99df00938cb561",
        files=dict((n, sha(ROOT / n)) for n in selected),
        data_root=DATA_ROOT.as_posix(), data_files=DATA_FILES,
        native_sources={n: row["sha256"] for n, row in actual["files"].items()},
        versions=actual["versions"],
        config_path="docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml",
        phase_config_path="coco_s4/configs/final_report.json", phase_dependencies=observation["dependencies"])
    local = EX / tag / "final_report_release.json"
    write_once(local, release)
    return local


def prepare(tag):
    tag = source_tag(tag)
    _, transport = services(); observation = read_dependency(transport)
    if observation.get("state") != "READY_FOR_FINAL_REPORT":
        print(json.dumps(observation)); return
    head = current_source(); local = build(tag, head, observation)
    print(json.dumps(dict(prepared=str(local), source_commit=head, uploaded=False, registered=False)))


def register(tag):
    tag = source_tag(tag)
    spec_path = EX / (tag + ".json")
    release_path = EX / tag / "final_report_release.json"
    if not spec_path.is_file() or not release_path.is_file():
        raise ValueError("Final report release was not prepared")
    spec = load(spec_path); release = load(release_path)
    if current_source() != release["source_commit"] or spec["source"]["commit"] != release["source_commit"]:
        raise ValueError("Final-report source no longer matches its committed source tree")
    _, transport = services(); observation = read_dependency(transport)
    if (observation.get("state") != "READY_FOR_FINAL_REPORT"
            or observation.get("dependencies") != release["phase_dependencies"]):
        raise ValueError("Accepted receipts or fresh resource gate changed; no registration")
    remote_root = spec["remote"]["repo_root"]
    remote_release = remote_root + "/final_report_release.json"
    remote_ready = (WORKFLOW_OUTPUT / "receipts/final_report_READY.json").as_posix()
    ready_path = EX / tag / "final_report_READY.json"
    ready = dict(phase="final_report", workflow_run_id=WORKFLOW, pre_run_result="pass",
        source_root=remote_root, release_sha256=sha(release_path), release_relative="final_report_release.json",
        module="coco_s4.phase_final_report",
        previous_stage_receipt_sha256=release["phase_dependencies"]["official_eval"]["accepted_sha256"],
        registered_UTC=now(), review_not_applicable="deterministic post-evaluation report; no model execution or metrics")
    if ready_path.exists():
        old = load(ready_path)
        if any(old.get(k) != v for k, v in ready.items() if k != "registered_UTC") or not old.get("registered_UTC"):
            raise ValueError("Retained final-report registration differs; no overwrite")
        ready = old
    else:
        write_once(ready_path, ready)
    inspector = r'''import hashlib,json,sys
from pathlib import Path
out={}
for name in sys.argv[1:]:
 p=Path(name)
 if p.exists():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
  out[name]=h.hexdigest()
 else:out[name]=None
print(json.dumps(out,sort_keys=True))'''
    remote = transport.run([PYTHON, "-B", "-c", inspector, remote_release, remote_ready])
    if remote.returncode:
        raise ValueError("Could not preflight final-report release/READY paths")
    found = json.loads(remote.stdout)
    wanted = {remote_release: sha(release_path), remote_ready: sha(ready_path)}
    if any(found.get(p) not in (None, digest) for p, digest in wanted.items()):
        raise ValueError("Conflicting report release/READY exists; no overwrite or duplicate registration")
    for remote_path, local_path in ((remote_release, release_path), (remote_ready, ready_path)):
        if found.get(remote_path) is None:
            transport.upload(local_path, remote_path, wanted[remote_path])
        check = transport.run([PYTHON, "-B", "-c", inspector, remote_path])
        if check.returncode or json.loads(check.stdout).get(remote_path) != wanted[remote_path]:
            raise ValueError("Final-report registration readback mismatch: " + remote_path)
    print(json.dumps(dict(phase="final_report", registered_existing_controller=True,
                          READY_uploaded_and_readback=True, serial_controller_owns_report_start=True,
                          new_scheduler=False, GPU_work=0)))


def main():
    p = argparse.ArgumentParser(); p.add_argument("action", choices=("prepare", "register")); p.add_argument("--tag", required=True)
    a = p.parse_args(); {"prepare": prepare, "register": register}[a.action](a.tag)


if __name__ == "__main__": main()
