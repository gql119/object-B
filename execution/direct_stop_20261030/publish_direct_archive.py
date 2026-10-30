"""Explicit small code/evidence archive only; no data/weights or whole-tree add."""
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('action',choices=['commit','push']);args=q.parse_args()
    if args.action=='push':
        v=json.loads((HERE/'ARCHIVE_COMMIT.json').read_text());assert git('rev-parse','HEAD').decode().strip()==v['archive_commit']
        r=subprocess.run(['git','push','origin','HEAD:refs/heads/codex/s4-at-robustness-20261027'],cwd=ROOT,capture_output=True,text=True)
        (HERE/'ARCHIVE_PUSH_ATTEMPT.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,archive_commit=v['archive_commit']),indent=2),encoding='utf8')
        print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,archive_commit=v['archive_commit'])));raise SystemExit(r.returncode)
    assert not git('diff','--cached','--name-only').strip(),'Preserve unrelated staged changes'
    names=['.gitattributes','update_direct_progress.py','finite_remaining_samples.py','finish_fallback_preflight.py','finish_remaining_preflight.py',
        'verified_api_preflight_client.py','observe_current_phase.py','build_fallback_review.py','close_and_upload_fallback.py','write_direct_report.py',
        'observe_new_formal.py','record_fallback_decision.py','finalize_readiness.py','record_new_formal.py','publish_direct_archive.py',
        'REMOTE_BRANCH_API_VERIFIED.json','CURRENT_RUNTIME_CPU_REUSE.json','CURRENT_GPU_MAIN_AUDIT.json','CURRENT_GT_AND_BUDGET.json',
        'SELECTED_PROTOCOL_CHANGE_ROUTE.json','SELECTED_PROTOCOL_SCIENTIFIC_REVIEW_PACKET.json','SELECTED_PROTOCOL_PACKET_READY.json',
        'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW.json','SELECTED_PROTOCOL_SCIENTIFIC_REVIEW.md','SELECTED_PROTOCOL_SCIENTIFIC_MAIN_CLOSURE.json',
        'SELECTED_ACCEPTED_PREFLIGHT.json','SELECTED_GATE_UPLOAD_VERIFIED.json','SELECTED_RUNTIME.json','NEW_FORMAL_OBSERVATION.json',
        'FALLBACK_SELECTION_PENDING_GATES.json','ACTUAL_CUDA_COMPARISON_TABLE.json','ENGINEERING_IMPLEMENTATION_SELECTION.json',
        'ABBA_CANDIDATE_02_B36.json','ABBA_CANDIDATE_03_B36.json','BATCH_ABBA_Coriginal_B72_PLAN.json','BATCH_ABBA_Coriginal_B72_ASSESSMENT.json',
        'BASELINE_PREPARATION_TLS_FAILURE.json']
    paths=[HERE/n for n in names]+[ROOT/'docs/S4_AT_ROBUSTNESS_20261027'/n for n in ['CURRENT_STATE.json','S4_AT_DIRECT_STOP_CUDA_OPTIMIZATION_AND_RESTART_REPORT.md']]+[ROOT/'execution/AT_CONTINUE.md']
    for tag in ['s4_at_selected_native_20261030_gpu01','s4_at_selected_baseline_20261030_gpu01','s4_at_selected_phase_20261030_gpu01']:
        paths += [HERE/'current_probes'/tag/n for n in ['runspec.json','READY.json','launch.json','completion_health.json','wait.json','pull.json','summary.json']]
    for label in ['eq02c4','eq03c4','bo1v2','b21','b22','bo2','bo3','b31','b32','bo4','bcoriginal_72_b36r1','bcoriginal_72_b72r1','bcoriginal_72_b72r2','bcoriginal_72_b36r2']:
        paths.append(HERE/'probes'/label/'summary.json')
    tag='s4_at_serial_seed0_20261030_fresh_v1'
    paths += [HERE/tag/n for n in ['runspec.json','prepare_capacity.json','READY.json','launch_intent.json','launch.json']]
    records=[]
    for p in paths:
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Unexpected archive path: '+str(p))
        if p.stat().st_size>1024*1024 or p.suffix.lower() in ('.pt','.png','.jpg','.zip','.npy','.npz'):raise ValueError('Not a small allowed code/report file: '+str(p))
        records.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    manifest=HERE/'ARCHIVE_MANIFEST.json';manifest.write_text(json.dumps(dict(runtime_commit='87456fa7a57206551b0e3be0253cc4866f0cdcf1',
        scope='Explicit small source/reports only; no data/images/weights/profiles/secrets',files=records),indent=2),encoding='utf8')
    paths.append(manifest);rels=sorted({p.relative_to(ROOT).as_posix() for p in paths})
    subprocess.run(['git','-c','core.autocrlf=false','add','-f','--',*rels],cwd=ROOT,check=True)
    staged=set(git('diff','--cached','--name-only').decode().splitlines())
    if not staged or staged-set(rels):raise ValueError('Unexpected staged scope')
    result=subprocess.run(['git','commit','-m','docs: archive real CUDA comparisons and fresh AT fallback launch'],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr+result.stdout)
    commit=git('rev-parse','HEAD').decode().strip()
    for rel in rels:
        if git('show',commit+':'+rel)!=(ROOT/rel).read_bytes():raise ValueError('Git raw blob differs from retained local evidence: '+rel)
    proof=dict(archive_commit=commit,runtime_commit='87456fa7a57206551b0e3be0253cc4866f0cdcf1',file_count=len(rels),bytes=sum((ROOT/r).stat().st_size for r in rels),
        all_scoped_Git_blob_bytes_exact=True,independent_remote_blob_downloads_claimed=False,remote_ref_verification_pending=True,scoped_paths=rels)
    (HERE/'ARCHIVE_COMMIT.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in proof.items() if k!='scoped_paths'}))
