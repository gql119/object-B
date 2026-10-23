"""Local deterministic config builder from this conversation's actual identity audits."""
from pathlib import Path
import json,hashlib,subprocess
W=Path(__file__).resolve().parents[1];B=W.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    if list((W/'execution/state').glob('*/run_ref.json')):
        raise ValueError('Deployed bindings exist; use a new source-bound queue configuration, never regenerate a live queue')
    old=read(W/'docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_LOCK.json')
    audit=read(W/'execution/SERVER_READONLY_AUDIT.json');full=read(B/'s4_tfss_full_overlap_20261021/execution/S4_TFSS_FULL_CURRENT_STATE.json')
    remote=old['remote_root'].rsplit('/',1)[0]+'/robustness_s4_extended_20261022'
    cfg=W/'robustness_s4_extended/configs'
    configs={
      'cutout':dict(name='cutout',n_holes=1,area_fraction=[.25,.25],probability=1.,fill_value=0,max_gt_occlusion=.5,max_attempts=8,occlusion_policy='reject whole candidate if any GT above .5; retain all original full boxes; skip after8; all20classes identical',schedule='all200epochs; native Mosaic closes after100'),
      'cutmix':dict(name='cutmix',probability=1.,beta=1.,num_areas=3,donor_min_visible=.5,min_pixels=2.,primary_occlusion='native zero GT intersection required',schedule='native close_mosaic100 disables CutMix for final100epochs',base='native CutMix with stricter donor filtering'),
      'mixup':dict(name='mixup',probability=1.,beta=[32.,32.],labels='native Instances and integer class concatenation; no per-instance lambda loss weights',schedule='native close_mosaic100 disables MixUp for final100epochs'),
      'op':dict(name='op',enabled=False,state='BLOCKED_PROTOCOL_APPROVAL',OP_PROTOCOL_APPROVED=False,selected_protocol=None,recommendation='B pooled class-agnostic instance basis at64x64; overlap simultaneous averaging; proposal only')}
    for name,c in configs.items():write(cfg/(name.upper()+'_CONFIG.json'),c)
    q=dict(schema='s4.extended.queue.v1',pipeline_run_id='s4_robustness_extended_seed0_20261022_v1',root=remote,output_root=remote+'/outputs/s4_robustness_extended_seed0_20261022_v1',python='/project/guoqinglong/miniconda3/envs/cfbdm/bin/python',order=['cutout','cutmix','mixup','op'],poll_seconds=300,monitor_seconds=7200,isolate_job_scopes=True,AT_authorized=False,baseline=old,
        predecessor=dict(run_id=full['formal_run_id'],output=full['remote_root']+'/outputs/'+full['formal_run_id'],source_commit=full['runtime_scientific_commit'],config_sha256=full['config_sha256'],native_state='complete',required_adapter_state='COMPLETED_AND_ACCEPTED'),
        native_sources=audit['ultralytics_sources'],defenses=configs,smoke_batches_per_arm=20,smoke_state='PENDING_GPU_SMOKE',review_state='PENDING_GPU_SMOKE_THEN_ONE_SCIENTIFIC_REVIEW',review_receipt=remote+'/receipts/EXTENDED_SCIENTIFIC_CLOSURE.json',OP_PROTOCOL_APPROVED=False,source_files={})
    # Capture scientific and orchestration sources, plus all immutable baseline code they reuse.
    paths=list((W/'robustness_s4_extended').rglob('*.py'))+list((W/'ue_framework').rglob('*.py'))+list((W/'scripts').glob('*.py'))+[W/'configs/voc_yolov8n_20cls.yaml',W/'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json']
    for p in paths:
        if '__pycache__' not in str(p):
            rel=p.relative_to(W).as_posix()
            # RRCTL reconstructs Git blobs on Linux. Windows autocrlf working bytes are not that identity.
            blob=subprocess.check_output(['git','show','HEAD:'+rel],cwd=W)
            if blob.replace(b'\r\n',b'\n')!=p.read_bytes().replace(b'\r\n',b'\n'):
                raise ValueError('Commit scientific source before freezing it: '+rel)
            q['source_files'][rel]=hashlib.sha256(blob).hexdigest()
    q['source_hash_basis']='exact committed Git blob bytes; Linux RRCTL checkout; original canonical -text files preserved'
    write(cfg/'QUEUE.json',q)
    write(W/'robustness_s4_extended/reports/QUEUE_STATUS.json',dict(state='PREPARED_NOT_DEPLOYED',predecessor=audit['full_state'],GPU_training_started=False,stages={x:('BLOCKED_PROTOCOL_APPROVAL' if x=='op' else 'WAITING_PREVIOUS_PIPELINE') for x in q['order']},official_metrics=None))
    print(json.dumps(dict(configs=4,scientific_sources=len(q['source_files']),root=remote)))
if __name__=='__main__':main()
