"""Prepare, smoke, and register one paired official-evaluation candidate.

All formal starts remain owned by the already-running serial controller.
This helper creates one isolated bounded GPU validation job per unique tag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .phase_contract import OWNED, WORKFLOW, WORKFLOW_OUTPUT
from .dimension_protocol import PROTOCOL as DIMENSION_PROTOCOL
from .release_transport import anchor_path, ensure_remote_exact, validate_anchor_outside_lifecycle
from .runtime import DATA_FILES, DATA_ROOT, ROOT, load, now, sha, write_once

EX = ROOT / "execution"
PYTHON = "/project/guoqinglong/miniconda3/envs/cfbdm/bin/python"
PREDECESSORS = ("teacher200", "teacher_eval", "calibration", "fourier24", "export25k", "clean200", "s4_200")
CPU_BLOCKER_EVIDENCE = EX / "COCO_OFFICIAL_EVAL_BLOCKER_FIX_CPU_20261029_V2.json"
BLOCKER_SOURCE_PATHS = {
    "B1": ("coco_s4/evaluation.py", "coco_s4/phase_official_eval.py", "coco_s4/configs/official_eval.json"),
    "B2": ("coco_s4/coco_eval.py", "coco_s4/phase_official_eval.py"),
}

READ_DEPENDENCY = r'''
from pathlib import Path
import hashlib,json,os,sys
o=Path(sys.argv[1]);run=sys.argv[2];source=sys.argv[3]
s=json.loads((o/'pipeline_state.json').read_text())
assert os.getuid()==1014 and s['run_id']==run and s['source_commit']==source
ident=s['controller'];p=Path('/proc')/str(ident['pid']);f=(p/'stat').read_text().rsplit(')',1)[1].split()
assert p.stat().st_uid==1014 and int(f[19])==ident['start_ticks'] and f[0]!='Z'
if s.get('state')!='WAITING_PHASE_RELEASE' or s.get('current_stage')!='official_eval' or s.get('active_job') is not None or s.get('active_child') is not None:
 print(json.dumps(dict(state='WAITING_SERIAL_SLOT',native_state=s.get('state'),current_stage=s.get('current_stage'),active_job=s.get('active_job'))));sys.exit(0)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
deps={}
for name in ('teacher200','teacher_eval','calibration','fourier24','export25k','clean200','s4_200'):
 stage=s.get('stages',{}).get(name,{})
 path=o/name/'accepted.json';r=json.loads(path.read_text());digest=sha(path)
 assert stage.get('state')=='ACCEPTED' and stage.get('receipt_sha256')==digest and r.get('state')=='ACCEPTED'
 row=dict(accepted_path=str(path),accepted_sha256=digest)
 if name in ('clean200','s4_200'):
  ck=o/name/'train/weights/last.pt';full=o/name/'train/state_snapshots/epoch_200_full_state.pt'
  assert r.get('completed_epochs')==200 and r.get('primary_checkpoint')=='last.pt@200'
  assert r.get('zero_validation_forwards') is True and r.get('pretrained') is False
  assert sha(ck)==r['primary_checkpoint_sha256'] and sha(full)==r['full_state_sha256']
  row.update(checkpoint_path=str(ck),checkpoint_sha256=sha(ck),full_state_path=str(full),full_state_sha256=sha(full),initial_state_sha256=r['initial_state_sha256'])
 deps[name]=row
clean=json.loads((o/'clean200/accepted.json').read_text());s4=json.loads((o/'s4_200/accepted.json').read_text())
assert clean['initial_state_sha256']==s4['initial_state_sha256'] and s4['matched_clean_initial_state'] is True
te=json.loads((o/'teacher_eval/accepted.json').read_text())
assert te['image_count']==5000 and te['classes']==80 and te['parameter_updates'] is False
assert json.loads((o/'export25k/accepted.json').read_text())['labels_byte_identical'] is True
sys.path.insert(0,sys.argv[4]);from coco_s4.serial import capacity
cap=capacity(o)
teacher=o/'teacher200/accepted.json';tr=json.loads(teacher.read_text());last=o/'teacher200/train/weights/last.pt';full=o/'teacher200/train/state_snapshots/epoch_200_full_state.pt'
td=dict(accepted_path=str(teacher),accepted_sha256=sha(teacher),checkpoint_path=str(last),checkpoint_sha256=sha(last),full_state_path=str(full),full_state_sha256=sha(full))
print(json.dumps(dict(state='READY_FOR_ISOLATED_PHASE_VALIDATION' if cap.get('admitted') else 'WAITING_RESOURCES',
 phase='official_eval',dependencies=deps,teacher_dependency=td,capacity=cap,controller=ident)))
'''


def services():
    from .control import services as existing
    return existing()


def candidate_name(tag):
    if not re.fullmatch(r"coco_official_eval_[a-z0-9_]{1,36}", tag):
        raise ValueError("Use one unique coco_official_eval_... candidate tag")
    return tag


def current_source():
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError("Tracked computation source must be committed before official-evaluation validation")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(ROOT), "ls-remote", "origin",
        "refs/heads/codex/s4-coco-risk-fallback-20261027"], text=True).split()
    if not remote or remote[0] != head:
        raise ValueError("Official-evaluation candidate must be published and remotely verified")
    return head


def read_dependency(transport):
    from .workflow_admission import local_controller_binding
    controller = local_controller_binding()
    active = Path(controller['remote_repo_root'])
    r = transport.run([PYTHON, "-B", "-c", READ_DEPENDENCY, WORKFLOW_OUTPUT.as_posix(), WORKFLOW,
                       controller['source_commit'], active.as_posix()])
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace")[-3000:])
    return json.loads(r.stdout)


def build(tag, head, observation):
    if observation.get("state") != "READY_FOR_ISOLATED_PHASE_VALIDATION":
        raise ValueError("Serial predecessor chain or fresh capacity has not passed")
    actual = load(EX / "ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT_20261025.json")
    addon = load(EX / "ISOLATED_EVALUATOR_CPU_VALIDATION_20261025.json")
    names = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
    selected = [n for n in names if n.startswith(("coco_s4/", "ue_framework/", "configs/")) or n in
                ("docs/S4_COCO_MINITRAIN_20261023/COCO_CATEGORY_MAPPING.json",
                 "docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml")]
    required = ("coco_s4/phase_official_eval.py", "coco_s4/phase_contract.py",
                "coco_s4/configs/official_eval.json", "coco_s4/coco_eval.py")
    if any(n not in selected for n in required):
        raise ValueError("Official-evaluation source/configuration is not in the immutable source closure")
    release = dict(schema="s4.coco.release.v1", phase="official_eval", workflow_run_id=WORKFLOW,
                   source_commit=head, scientific_parent="b1e29510e7b1ed5fa0fe361a27132d8790e65c14",
                   victim_parent="ab626917f20917e58cf30033ac99df00938cb561",
                   files={n: sha(ROOT / n) for n in selected}, data_root=DATA_ROOT.as_posix(), data_files=DATA_FILES,
                   native_sources={n: r["sha256"] for n, r in actual["files"].items()}, versions=actual["versions"],
                   config_path="docs/S4_COCO_MINITRAIN_20261023/COCO_EXPERIMENT_A_CONFIG.yaml",
                   phase_config_path="coco_s4/configs/official_eval.json",
                   teacher_dependency=observation["teacher_dependency"], phase_dependencies=observation["dependencies"],
                   evaluator=dict(directory=addon["addon"], files=addon["files"], wheel_sha256=addon["wheel_sha256"]))
    release["dimension_protocol"] = DIMENSION_PROTOCOL
    local_release = EX / tag / "source_release.json"
    from .workflow_admission import attach
    attach(release)
    write_once(local_release, release)
    template = load(EX / (WORKFLOW + ".json"))
    remote_repo = (OWNED / "releases" / tag).as_posix()
    remote_output = (OWNED / "validation" / tag).as_posix()
    remote_release = anchor_path(tag)
    template.update(run_id=tag, project="s4-coco-paired-official-eval-validation")
    template["source"] = dict(repo_root=str(ROOT), branch="codex/s4-coco-risk-fallback-20261027",
                              commit=head, allowed_post_commit_paths=["execution/", "Ultralytics/"])
    template["remote"].update(repo_root=remote_repo, output_root=remote_output,
                              stage_root=(OWNED / "stage" / tag).as_posix(),
                              control_root=(OWNED / "control" / tag).as_posix())
    validate_anchor_outside_lifecycle(remote_release, template["remote"])
    template["session"]["name"] = tag
    unit = "s4coco-official-eval-" + hashlib.sha256(tag.encode()).hexdigest()[:20]
    template["workload"] = dict(cwd=".", argv=["systemd-run", "--user", "--scope", "--quiet", "--unit=" + unit,
        PYTHON, "-B", "-m", "coco_s4.process", "--job", remote_output + "/supervision", "--cwd", remote_repo, "--",
        PYTHON, "-B", "-m", "coco_s4.phase_official_eval", "--release", remote_release,
        "--output", remote_output + "/result", "--smoke-batches", "1"])
    template["anchors"] = [dict(name="phase_release", local_path=str(local_release), remote_path=remote_release,
                                sha256=sha(local_release))]
    adapter = "import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['output_root'])/'result/status.json';s=json.loads(p.read_text()) if p.exists() else {};print(json.dumps(dict(protocol='rrctl.adapter.v1',healthy=s.get('state')!='FAILED',complete=s.get('state')=='passed',progress=s,artifacts=['result/status.json','result/smoke_summary.json'])))"
    for health in template["health"].values():
        health["adapter_argv"] = [PYTHON, "-B", "-c", adapter]
        health["gpu_utilization_policy"] = "advisory"
        health["timeout_seconds"] = 900
    template["artifacts"] = [dict(path="result/" + name, required=True) for name in
        ("smoke_summary.json", "phase_spec.json", "status.json", "val2017_image_identity.json")]
    template["local_pull_root"] = str(EX / "p")
    template["metadata"] = dict(execution_purpose="isolated_paired_official_eval_pre_review_smoke",
        computation_kind="inference", candidate_commit=head, cleanup_policy="retain_all",
        official_metrics_disabled=True, artifact_ingest_disabled=True,
        selected_inputs="first 36 sorted val2017 IDs per matched checkpoint; all 5000 image bytes identity-checked",
        batches_per_checkpoint=1,
        max_total_raw_forwards=100, one_shot_no_scheduler=True, no_formal_phase_registration=True)
    from remote_run_control.models import RunSpec
    RunSpec.from_dict(template)
    spec_path = EX / (tag + ".json")
    write_once(spec_path, template)
    write_once(EX / tag / "dependency_observation.json", observation)
    return spec_path


def prepare(tag):
    tag = candidate_name(tag)
    if (EX / (tag + ".json")).exists() or (EX / "rrctl_state" / tag).exists() or (EX / tag).exists():
        raise FileExistsError("Prepared/bound evaluation candidate exists; inspect only")
    _, transport = services(); observation = read_dependency(transport)
    if observation.get("state") != "READY_FOR_ISOLATED_PHASE_VALIDATION":
        print(json.dumps(observation)); return
    head = current_source(); spec_path = build(tag, head, observation)
    print(json.dumps(dict(prepared=str(spec_path), candidate_commit=head, launched=False, formal_registered=False)))


def launch_smoke(tag):
    tag = candidate_name(tag); spec_path = EX / (tag + ".json")
    if (EX / "rrctl_state" / tag).exists() or (EX / tag / "launch_intent.json").exists():
        raise FileExistsError("Bound evaluation smoke is inspect-only; never relaunch")
    cfg = load(spec_path)
    if current_source() != cfg["source"]["commit"]:
        raise ValueError("Prepared evaluation source changed")
    ctl, transport = services(); observation = read_dependency(transport)
    if observation.get("state") != "READY_FOR_ISOLATED_PHASE_VALIDATION":
        print(json.dumps(observation)); return
    release = load(EX / tag / "source_release.json")
    if (observation["teacher_dependency"] != release["teacher_dependency"]
            or observation["dependencies"] != release["phase_dependencies"]):
        raise ValueError("Accepted dependencies changed after evaluation preparation")
    paths = [cfg["remote"][key] for key in ("repo_root", "output_root", "stage_root", "control_root")]
    check = transport.run([PYTHON, "-B", "-c",
        "import json,os,sys;from pathlib import Path;assert os.getuid()==1014;assert all(not Path(p).exists() for p in json.loads(sys.argv[1]));print('absent')",
        json.dumps(paths)])
    if check.returncode:
        raise ValueError("Existing remote candidate path; no launch")
    from remote_run_control.models import RunSpec
    spec = RunSpec.from_path(spec_path); readiness = ctl.ready(spec); write_once(EX / tag / "ready.json", readiness)
    if not readiness.get("ready"):
        raise ValueError("RRCTL rejected isolated evaluation smoke readiness")
    temp = EX / "rrctl_temp"; temp.mkdir(exist_ok=True); os.environ["TMP"] = os.environ["TEMP"] = str(temp)
    write_once(EX / tag / "launch_intent.json", dict(UTC=now(), run_id=tag, source_commit=cfg["source"]["commit"]))
    result = ctl.launch(spec); write_once(EX / tag / "launch.json", result)
    print(json.dumps(dict(launched_once=True, run_id=tag, result=result, formal_registered=False)))


def register(tag, review_path):
    tag = candidate_name(tag); spec_path = EX / (tag + ".json")
    if not spec_path.is_file() or not (EX / "rrctl_state" / tag).exists():
        raise ValueError("No bound RRCTL official-evaluation smoke")
    spec = load(spec_path)
    if current_source() != spec["source"]["commit"]:
        raise ValueError("Review source is no longer current")
    summary_path = Path(spec["local_pull_root"]) / tag / "result/smoke_summary.json"
    if not summary_path.is_file():
        raise ValueError("RRCTL has not pulled the isolated smoke result")
    smoke = load(summary_path); release_path = EX / tag / "source_release.json"
    release = load(release_path); cfg_sha = sha(ROOT / release["phase_config_path"])
    identity_path = summary_path.parent / "val2017_image_identity.json"
    if not identity_path.is_file():
        raise ValueError("Bounded smoke did not retain the full val2017 per-image identity receipt")
    image_identity = load(identity_path)
    expected_manifest_sha = DATA_FILES["metadata/VAL2017_IMAGE_MANIFEST_20261025.jsonl"]
    if (image_identity.get("state") != "passed" or image_identity.get("image_count") != 5000
            or image_identity.get("expected_image_count") != 5000
            or image_identity.get("manifest_sha256") != expected_manifest_sha
            or len(image_identity.get("verified_images", [])) != 5000
            or smoke.get("val2017_image_identity_receipt_sha256") != sha(identity_path)):
        raise ValueError("Bounded smoke val2017 identity receipt is incomplete or mismatched")
    if (smoke.get("state") != "passed" or smoke.get("phase") != "official_eval"
            or smoke.get("source_commit") != spec["source"]["commit"]
            or smoke.get("release_sha256") != sha(release_path)
            or smoke.get("phase_config_sha256") != cfg_sha
            or smoke.get("official_metrics_disabled") is not True
            or smoke.get("artifact_ingest_disabled") is not True
            or smoke.get("parameter_updates") is not False
            or smoke.get("checkpoint_selection_allowed") is not False
            or smoke.get("risk_estimator_policy") != release.get("risk_estimator_policy")
            or smoke.get("dimension_protocol") != release.get("dimension_protocol")
            or smoke.get("strict_Original_S4_equivalence_claim") is not False
            or smoke.get("val2017_image_manifest_sha256") != expected_manifest_sha
            or smoke.get("val2017_image_count") != 5000
            or set(smoke.get("victim_probes", {})) != {"clean200", "s4_200"}
            or not 1 <= smoke.get("total_raw_forwards_including_warmups", 0) <= 100):
        raise ValueError("Paired official-evaluation candidate smoke failed its current-release contract")
    for probe in smoke["victim_probes"].values():
        eq = probe.get("baseline_equivalence_probe", {})
        if probe.get("actual_device") != "cuda:0" or not probe.get("finite_outputs"):
            raise ValueError("GPU smoke did not produce finite output on CUDA0")
        if (eq.get("reference_actual_device") != probe.get("actual_device")
                or not eq.get("raw_outputs_exact") or not eq.get("decoded_predictions_exact")
                or eq.get("tolerance") != 0):
            raise ValueError("Candidate differs from independent native YOLO inference")
    from .phase_official_eval import (ORIGINAL_BLOCKER_IDS, RESOLVED_FINDING_IDS,
                                      validate_original_review)
    review = load(review_path)
    if validate_original_review(review) != ORIGINAL_BLOCKER_IDS:
        raise ValueError("The retained original independent review is not the expected blocking review")
    if not CPU_BLOCKER_EVIDENCE.is_file():
        raise ValueError("CPU blocker-closure evidence must pass on the current source before registration")
    cpu_validation = load(CPU_BLOCKER_EVIDENCE)
    required_source_paths = sorted({path for paths in BLOCKER_SOURCE_PATHS.values() for path in paths})
    source_file_sha256 = {path: release.get("files", {}).get(path) for path in required_source_paths}
    if (cpu_validation.get("schema") != "s4.coco.official-eval-blocker-fix-cpu.v1"
            or cpu_validation.get("state") != "passed"
            or cpu_validation.get("source_commit") != spec["source"]["commit"]
            or cpu_validation.get("source_file_sha256") != source_file_sha256
            or cpu_validation.get("result", {}).get("state") != "passed"
            or cpu_validation.get("result", {}).get("same_name_same_size_substitution_rejected") is not True
            or cpu_validation.get("result", {}).get("missing_metric_rejected") is not True
            or cpu_validation.get("result", {}).get("protocol_identity_contract_passed") is not True
            or cpu_validation.get("GPU_units") != 0
            or cpu_validation.get("official_metrics_generated") != 0
            or cpu_validation.get("formal_registration") is not False):
        raise ValueError("CPU blocker-fix evidence is not bound to this current source candidate")
    _, transport = services(); observation = read_dependency(transport)
    if (observation.get("state") != "READY_FOR_ISOLATED_PHASE_VALIDATION"
            or observation.get("dependencies") != release["phase_dependencies"]):
        raise ValueError("Existing serial phase gate/dependencies/capacity changed; no registration")
    reviewer_local = Path(review_path).resolve()
    receipt_root = WORKFLOW_OUTPUT / "receipts"
    remote_review = (receipt_root / "official_eval_ORIGINAL_NEGATIVE_REVIEW.json").as_posix()
    remote_cpu = (receipt_root / "official_eval_BLOCKER_FIX_CPU.json").as_posix()
    remote_blocker = (receipt_root / "official_eval_BLOCKER_CLOSURE.json").as_posix()
    remote_closure = (receipt_root / "official_eval_SCIENTIFIC_CLOSURE.json").as_posix()
    remote_ready = (receipt_root / "official_eval_READY.json").as_posix()
    blocker_path = EX / tag / "blocker_closure.json"
    closure_path = EX / tag / "scientific_closure.json"
    registration_path = EX / tag / "phase_registration.json"
    remote_smoke = spec["remote"]["output_root"] + "/result/smoke_summary.json"
    remote_identity = spec["remote"]["output_root"] + "/result/val2017_image_identity.json"
    cpu_file_sha256 = sha(CPU_BLOCKER_EVIDENCE)
    blocker_core = dict(
        schema="s4.coco.official-eval-blocker-closure.v1", state="closed", phase="official_eval",
        original_review_id=review["review_id"], original_reviewed_source_commit=review["pre_run_code_commit"],
        original_review_sha256=sha(reviewer_local), candidate_source_commit=spec["source"]["commit"],
        release_sha256=sha(release_path), phase_config_sha256=cfg_sha,
        smoke_sha256=sha(summary_path), cpu_validation_path=remote_cpu,
        cpu_validation_sha256=cpu_file_sha256,
        val2017_image_manifest_sha256=expected_manifest_sha,
        val2017_image_identity_receipt_sha256=sha(identity_path),
        closed_blocker_ids=list(ORIGINAL_BLOCKER_IDS), closed_attribution_gap_ids=["G1"],
        unresolved_blocker_ids=[], source_file_sha256=source_file_sha256,
        findings={
            "B1": dict(state="closed", metric="absolute_clean_harm",
                       formula="clean Person AP50 - S4 Person AP50 (unclipped AP fraction)",
                       cpu_contract_passed=True, paired_metric_config_declared=True,
                       bounded_smoke_current_source_passed=True),
            "B2": dict(state="closed", identity_image_count=5000,
                       image_manifest_sha256=expected_manifest_sha,
                       cpu_same_name_same_size_substitution_rejected=True,
                       smoke_identity_receipt_sha256=sha(identity_path)),
            "G1": dict(state="closed", risk_estimator_policy=release["risk_estimator_policy"],
                       dimension_protocol=release["dimension_protocol"],
                       strict_Original_S4_equivalence_claim=False,
                       smoke_attribution_validated=True),
        })
    if blocker_path.exists():
        blocker_closure = load(blocker_path)
        if any(blocker_closure.get(k) != v for k, v in blocker_core.items()) or not blocker_closure.get("closed_UTC"):
            raise ValueError("Retained blocker closure differs from current source/smoke/evidence")
    else:
        blocker_closure = dict(blocker_core, closed_UTC=now(), closure_authority="main_agent_after_evidence")
        write_once(blocker_path, blocker_closure)
    closure_core = dict(phase="official_eval", source_commit=spec["source"]["commit"],
        release_sha256=sha(release_path), phase_config_sha256=cfg_sha,
        smoke_path=remote_smoke, smoke_sha256=sha(summary_path),
        blocker_closure_path=remote_blocker, blocker_closure_sha256=sha(blocker_path),
        cpu_validation_path=remote_cpu, cpu_validation_sha256=cpu_file_sha256,
        original_review_id=review["review_id"], original_reviewed_source_commit=review["pre_run_code_commit"],
        original_review_sha256=sha(reviewer_local),
        risk_estimator_policy=release["risk_estimator_policy"],
        dimension_protocol=release["dimension_protocol"],
        strict_Original_S4_equivalence_claim=False,
        val2017_image_manifest_sha256=expected_manifest_sha,
        val2017_image_identity_receipt_sha256=sha(identity_path), val2017_image_count=5000,
        review_path=remote_review, review_sha256=sha(reviewer_local), reviewer_count=1,
        pre_run_result="pass", reviewer_independence=True,
        independent_scientific_verdict="scientifically_incorrect",
        closure_verdict="blockers_resolved_by_main_agent",
        resolved_blocker_ids=list(ORIGINAL_BLOCKER_IDS), resolved_attribution_gap_ids=["G1"],
        unresolved_blockers=[])
    if closure_path.exists():
        closure = load(closure_path)
        if any(closure.get(k) != v for k, v in closure_core.items()) or not closure.get("closed_UTC"):
            raise ValueError("Retained closure differs from the exact reviewed candidate")
    else:
        closure = dict(closure_core, closed_UTC=now())
        write_once(closure_path, closure)
    ready_core = dict(phase="official_eval", workflow_run_id=WORKFLOW, pre_run_result="pass",
        source_root=spec["remote"]["repo_root"], release_sha256=sha(release_path),
        release_relative="source_release.json", module="coco_s4.phase_official_eval",
        previous_stage_receipt_sha256=release["phase_dependencies"]["s4_200"]["accepted_sha256"],
        scientific_closure_path=remote_closure, scientific_closure_sha256=sha(closure_path),
        smoke_path=remote_smoke, smoke_sha256=sha(summary_path),
        blocker_closure_path=remote_blocker, blocker_closure_sha256=sha(blocker_path),
        cpu_validation_path=remote_cpu, cpu_validation_sha256=cpu_file_sha256,
        original_review_path=remote_review, original_review_sha256=sha(reviewer_local),
        reviewer_count=1, reviewer_independence=True,
        independent_scientific_verdict="scientifically_incorrect",
        closure_verdict="blockers_resolved_by_main_agent",
        resolved_blocker_ids=list(ORIGINAL_BLOCKER_IDS), resolved_attribution_gap_ids=["G1"],
        risk_estimator_policy=release["risk_estimator_policy"],
        dimension_protocol=release["dimension_protocol"],
        strict_Original_S4_equivalence_claim=False,
        val2017_image_manifest_sha256=expected_manifest_sha,
        val2017_image_identity_receipt_sha256=sha(identity_path), val2017_image_count=5000)
    if registration_path.exists():
        ready = load(registration_path)
        if any(ready.get(k) != v for k, v in ready_core.items()) or not ready.get("registered_UTC"):
            raise ValueError("Retained READY differs from the exact reviewed candidate")
    else:
        ready = dict(ready_core, registered_UTC=now())
        write_once(registration_path, ready)
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
    remote_state = transport.run([PYTHON, "-B", "-c", inspector, remote_smoke, remote_identity,
                                  remote_review, remote_cpu, remote_blocker, remote_closure, remote_ready])
    if remote_state.returncode:
        raise ValueError("Could not inspect existing official-evaluation receipts")
    existing = json.loads(remote_state.stdout)
    if existing.get(remote_smoke) != sha(summary_path):
        raise ValueError("RRCTL smoke pull does not match the retained remote smoke bytes")
    if existing.get(remote_identity) != sha(identity_path):
        raise ValueError("RRCTL val2017 identity receipt differs from the retained local artifact")
    wanted = {remote_review: sha(reviewer_local), remote_cpu: cpu_file_sha256,
              remote_blocker: sha(blocker_path), remote_closure: sha(closure_path),
              remote_ready: sha(registration_path)}
    if any(existing.get(p) not in (None, digest) for p, digest in wanted.items()):
        raise ValueError("Conflicting serial phase receipt exists; no overwrite or duplicate registration")
    ensure_remote_exact(transport, release_path, spec["remote"]["repo_root"] + "/source_release.json")
    local_uploads = {remote_review: reviewer_local, remote_cpu: CPU_BLOCKER_EVIDENCE,
                     remote_blocker: blocker_path, remote_closure: closure_path,
                     remote_ready: registration_path}
    for remote_path, local_path in local_uploads.items():
        if existing.get(remote_path) is None:
            transport.upload(local_path, remote_path, wanted[remote_path])
        check = transport.run([PYTHON, "-B", "-c", inspector, remote_path])
        if check.returncode or json.loads(check.stdout).get(remote_path) != wanted[remote_path]:
            raise ValueError("Official-evaluation receipt upload failed exact readback: " + remote_path)
    print(json.dumps(dict(phase="official_eval", reviewer_count=1,
                          independent_scientific_verdict="scientifically_incorrect",
                          closure_verdict="blockers_resolved_by_main_agent", registered_existing_controller=True,
                          READY_uploaded_and_readback=True, serial_controller_owns_formal_start=True,
                          no_new_controller=True)))


def main():
    p = argparse.ArgumentParser(); p.add_argument("action", choices=("prepare", "launch-smoke", "register"))
    p.add_argument("--tag", required=True); p.add_argument("--review", type=Path)
    a = p.parse_args()
    if a.action == "register" and a.review is None:
        p.error("register requires --review JSON")
    {"prepare": prepare, "launch-smoke": launch_smoke, "register": register}[a.action](a.tag, *(([a.review] if a.action == "register" else [])))


if __name__ == "__main__": main()
