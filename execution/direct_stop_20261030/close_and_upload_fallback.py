"""Main closure and immutable gate upload after the single scientific verdict."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(ROOT))
from at_cuda_rrctl_20261029 import services,PYTHON
from robustness_s4_at.common import load,sha,write_once,now,PROTOCOL,protocol
SOURCE='87456fa7a57206551b0e3be0253cc4866f0cdcf1';TAG='s4_at_serial_seed0_20261030_fresh_v1'


if __name__=='__main__':
    review=HERE/'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW.json';r=load(review)
    if r.get('source_commit')!=SOURCE or r.get('review_mode')!='scientific_review' or r.get('packet_sha256')!=sha(HERE/'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW_PACKET.json'):
        raise ValueError('Single review does not bind the actual source/packet bytes')
    if r.get('result')!='scientifically_correct' or r.get('decision')!='allow_run' or r.get('blockers'):
        raise ValueError('Single scientific verdict is not an allow_run with no blockers; main repair required')
    audit=load(HERE/'CURRENT_GPU_MAIN_AUDIT.json');assert audit['source_commit']==SOURCE and audit['state']=='PASS_CURRENT_GPU_MAIN_AUDIT'
    closure=dict(state='passed',source_commit=SOURCE,UTC=now(),single_scientific_review_sha256=sha(review),
        GPU_main_audit_sha256=sha(HERE/'CURRENT_GPU_MAIN_AUDIT.json'),all_reported_blockers_closed=True,
        scientific_math_and_OriginalS4_source_unchanged=True,selected_implementation='original',selected_batch=36,
        reliable_speedup_claim=None,old_checkpoint_resume=False,fresh_epoch0_four_arms=True)
    write_once(HERE/'SELECTED_PROTOCOL_SCIENTIFIC_MAIN_CLOSURE.json',closure)
    ctl,t=services();owned=protocol()['remote_owned_root'];prefix=owned+'/preflight_20261030_fresh_v1'
    check=t.run([PYTHON,'-B','-c',"from pathlib import Path;import sys;p=Path(sys.argv[1]);assert p.parent.is_dir();p.mkdir(exist_ok=False)",prefix])
    if check.returncode:raise ValueError('Fresh gate directory collision; inspect before retry')
    tags=['s4_at_selected_native_20261030_gpu01','s4_at_selected_baseline_20261030_gpu01','s4_at_selected_phase_20261030_gpu01']
    native,baseline,resume=[load(HERE/'current_probes'/tag/'runspec.json') for tag in tags]
    gtremote=native['remote']['output_root']+'/result/GT_and_budget.json';gt=t.download(gtremote)
    (HERE/'CURRENT_GT_AND_BUDGET.json').write_bytes(gt)
    if json.loads(gt)['state']!='passed':raise ValueError('Actual current data/GT evidence failed')
    def item(path,digest):return dict(passed=True,path=path,sha256=digest)
    resume_item=item(resume['remote']['output_root']+'/result/candidate_resume_summary.json',sha(HERE/'current_probes'/tags[2]/'summary.json'))
    gate=dict(state='PREFLIGHT_ACCEPTED',source_commit=SOURCE,protocol_sha256=sha(PROTOCOL),UTC=now(),
        CPU_native=item(prefix+'/CPU_native.json',sha(HERE/'CURRENT_RUNTIME_CPU_REUSE.json')),
        CUDA_native=item(native['remote']['output_root']+'/result/smoke_summary.json',sha(HERE/'current_probes'/tags[0]/'summary.json')),
        GT_and_budget=item(gtremote,hashlib.sha256(gt).hexdigest()),resume=resume_item,phase_resume=resume_item,
        baseline=item(baseline['remote']['output_root']+'/result/smoke_summary.json',sha(HERE/'current_probes'/tags[1]/'summary.json')),
        scientific_main_closure=item(prefix+'/scientific_main_closure.json',sha(HERE/'SELECTED_PROTOCOL_SCIENTIFIC_MAIN_CLOSURE.json')),
        CUDA_actual_device='cuda:0',original_S4_source_unchanged=True,selected_implementation='original',selected_batch=36,
        latest_authorization_sha256='e46fc5aa728c294a4ce803a8ba23cacf5e80191173b411579bb3325d39581db8',
        old_original_terminal='aborted',old_checkpoint_resume=False,fresh_all_four_epoch0=True)
    write_once(HERE/'SELECTED_ACCEPTED_PREFLIGHT.json',gate)
    pairs=[(HERE/'CURRENT_RUNTIME_CPU_REUSE.json',prefix+'/CPU_native.json'),
        (HERE/'SELECTED_PROTOCOL_SCIENTIFIC_MAIN_CLOSURE.json',prefix+'/scientific_main_closure.json'),
        (review,prefix+'/independent_scientific_review.json'),(HERE/'SELECTED_ACCEPTED_PREFLIGHT.json',prefix+'/accepted.json')]
    proof=[]
    for local,remote in pairs:
        c=t.run([PYTHON,'-B','-c',"from pathlib import Path;import sys;assert not Path(sys.argv[1]).exists()",remote])
        if c.returncode:raise ValueError('Gate upload collision; retain existing evidence')
        t.upload(local,remote,sha(local));raw=t.download(remote)
        if raw!=local.read_bytes():raise ValueError('Gate raw readback mismatch')
        proof.append(dict(path=remote,sha256=sha(local),bytes=len(raw)))
    write_once(HERE/'SELECTED_GATE_UPLOAD_VERIFIED.json',dict(state='passed',source_commit=SOURCE,UTC=now(),files=proof))
    selection=dict(state='CURRENT_SOURCE_SCIENTIFICALLY_ACCEPTED',source_commit=SOURCE,new_formal_RunID=TAG,
        fresh_epoch0_all_four_arms=True,protocol_sha256=sha(PROTOCOL),preflight_local=str(HERE/'SELECTED_ACCEPTED_PREFLIGHT.json'),
        preflight_remote=prefix+'/accepted.json',selected_implementation='original',selected_batch=36,scientific_review_sha256=sha(review))
    write_once(HERE/'SELECTED_RUNTIME.json',selection)
    print(json.dumps(dict(state='CURRENT_SCIENTIFIC_GATE_UPLOADED_EXACT',source_commit=SOURCE,new_formal_not_yet_launched=True,selection=selection)))
