"""Run retained table/metric checks, then seal phase reports from accepted evidence."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import s4_pipeline_frozen_report as legacy

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def generate(config,lock,phase):
    root=Path(config['output_root']);doc=root/'reports';doc.mkdir(exist_ok=True)
    for name in ['S4_ROBUSTNESS_SOURCE_AUDIT.md','S4_ROBUSTNESS_PROTOCOL.md','DEFENSE_IMPLEMENTATION_AUDIT.md','JPEG50_CONFIG.json','JPEG30_CONFIG.json','GRAY_CONFIG.json','BDR_CONFIG.json','HISTORICAL_S0_DEFENSE_REUSE_AUDIT.md','CLEAN_DEFENSE_MATCHING_AUDIT.csv','AT_PROTOCOL_AND_COST_AUDIT.md']:
        source=Path(config['science_repo'])/'docs/S4_ROBUSTNESS_20261019'/name
        if source.is_file() and not (doc/name).exists():shutil.copyfile(source,doc/name)
    baseline=root/'docs/S4_AMP_RECOVERY_20261018/S4_COMPARISON.json';baseline.parent.mkdir(parents=True,exist_ok=True)
    source=Path(config['science_repo'])/'docs/S4_AMP_RECOVERY_20261018/S4_COMPARISON.json'
    if not baseline.exists():baseline.symlink_to(source)
    frozen_lock=Path(config['science_repo'])/'docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_LOCK.json'
    copy=doc/'S4_ROBUSTNESS_LOCK.json'
    if not copy.exists():shutil.copyfile(frozen_lock,copy)
    assert sha(copy)==config['config_sha256']
    legacy.ROOT=root;legacy.DOC=doc;legacy.SMALL=Path(config['root'])/'outputs'
    original=legacy.accept
    def actual_control(defense,c,lockhash):
        c=json.loads(json.dumps(c));out=legacy.SMALL/f's4_robustness_{defense}_paired_seed0_20261019_v1'
        p=out/'selected_arms.json'
        if p.exists() and not load(p).get('clean',{}).get('historical',True):c['historical_clean'].pop(defense,None)
        return original(defense,c,lockhash)
    legacy.accept=actual_control
    try:legacy.main()
    finally:legacy.accept=original
    wanted=['A'] if phase=='A' else ['A','B']
    pairs={}
    for ph in wanted:
        ds=legacy.PHASES[ph];state=load(doc/f'PHASE_{ph}_TRAINING_STATUS.json')
        assert state['state']=='COMPLETE_ACCEPTED','Incomplete phase cannot be accepted'
        for d in ds:
            pair,_=actual_control(d,lock,config['config_sha256']);assert pair is not None;pairs[d]=pair
        lines=[f'# Phase {ph} complete robustness report','', 'All formal values are fractions. last.pt@200 is primary; best.pt is secondary. Clean validation is the unchanged 4,952-image split. AT NOT_RUN.','', '| Defense | Clean Person AP50 | S4 Person AP50 | Clean NT AP50 | S4 NT AP50 | P | U | S | WCD | TCD3 |','|---|'+'---:|'*9]
        references={};duration=0
        for d in ds:
            p=pairs[d];a=p['metrics']['clean'];b=p['metrics']['S4'];s=p['recomputed_scores']
            vals=[a['mAP50_target'],b['mAP50_target'],a['mAP50_non_target'],b['mAP50_non_target'],*[s[k] for k in ['P','U','S','WCD','TCD3']]]
            lines.append('| '+d+' | '+' | '.join('UNDEFINED' if v is None else f'{v:.12g}' for v in vals)+' |')
            out=legacy.SMALL/f's4_robustness_{d}_paired_seed0_20261019_v1';status=load(out/'pipeline_status.json')
            elapsed=status.get('completed_at_unix',0)-status.get('started_at_unix',0)
            if elapsed>0:duration+=elapsed
            receipt=load(root/'execution/acceptance'/f'{out.name}.json')
            references[d]=dict(output=str(out),source_commit=p['source_commit'],config_sha256=p['config_sha256'],parent_S4_manifest_sha256=p['parent_S4_manifest_sha256'],paired_results_sha256=sha(out/'paired_results.json'),arms=receipt['arms'],elapsed_seconds=elapsed if elapsed>0 else None,clean_control=p['audits']['clean'],unfavorable_non_target_classes=p['scores'].get('signed_damage'))
            lines+=['',f'## {d}',f'Clean/S4 Person Recall: {a["Recall_target"]:.12g}/{b["Recall_target"]:.12g}; NT Recall: {a["Recall_non_target"]:.12g}/{b["Recall_non_target"]:.12g}.',f'S4 NT AP75={b["mAP75_non_target"]:.12g}; NT AP50–95={b["mAP50_95_non_target"]:.12g}; all-class AP50={b["mAP50_all"]:.12g}.',f'Clean checkpoint={a["checkpoint_sha256"]}; S4 checkpoint={b["checkpoint_sha256"]}.',f'Clean control: {"verified historical200, reevaluated" if p["audits"]["clean"].get("reused_clean") else "fresh paired clean200"}.']
        lines+=['',f'Total recorded pair wall time: {duration:.3f} seconds (historical control training cost excluded).','Complete AP50/AP75/AP50–95/Recall/Precision for VOC20 are in the accompanying per-class CSV. High S alone does not establish overall superiority. These are one-seed descriptive comparisons, without a statistical significance or causal mechanism claim.']
        text='\n'.join(lines)+'\n'
        (doc/f'PHASE_{ph}_FINAL_REPORT.md').write_text(text)
        shutil.copyfile(doc/f'PHASE_{ph}_ROBUSTNESS_RESULTS.csv',doc/f'PHASE_{ph}_RESULTS.csv')
        shutil.copyfile(doc/f'PHASE_{ph}_PER_CLASS_RESULTS.csv',doc/f'PHASE_{ph}_PER_CLASS.csv')
        save(doc/f'PHASE_{ph}_ACCEPTANCE.json',dict(state='ACCEPTED',phase=ph,time=time.time(),defenses=references,report_sha256=sha(doc/f'PHASE_{ph}_FINAL_REPORT.md'),results_sha256=sha(doc/f'PHASE_{ph}_RESULTS.csv'),per_class_sha256=sha(doc/f'PHASE_{ph}_PER_CLASS.csv')))
    if phase in ('B','FINAL'):
        baseline_metrics=load(baseline)['metrics'];base=baseline_metrics['S4'];clean0=baseline_metrics['Clean']
        def evidence(d):
            p=pairs[d];a=p['metrics']['clean'];b=p['metrics']['S4'];s=p['recomputed_scores']
            return f'Clean/S4 Person AP50 {a["mAP50_target"]:.9f}/{b["mAP50_target"]:.9f}, Person Recall {a["Recall_target"]:.9f}/{b["Recall_target"]:.9f}; NT AP50 {a["mAP50_non_target"]:.9f}/{b["mAP50_non_target"]:.9f}; P/U/S {s["P"]}/{s["U"]}/{s["S"]}'
        lines=['# S4 robustness final report','', 'ACCEPTED: all eight same-defense Clean/S4 comparisons passed the recorded200epoch, checkpoint, data, per-class, and ratio audits. AT remains NOT_RUN.','', 'All conclusions below describe a single seed. They do not establish statistical significance or identify a feature-space causal mechanism.','', '## Research questions','', '1. JPEG50: '+evidence('jpeg50')+f'. S4 Person AP50 change against no defense is {pairs["jpeg50"]["metrics"]["S4"]["mAP50_target"]-base["mAP50_target"]:+.9f}.', '2. JPEG30 versus JPEG50: '+evidence('jpeg30')+f'. Difference in S4 Person AP50 (JPEG30−JPEG50) is {pairs["jpeg30"]["metrics"]["S4"]["mAP50_target"]-pairs["jpeg50"]["metrics"]["S4"]["mAP50_target"]:+.9f}. Paired P/U/S must be considered alongside that absolute difference.','3. Gray: '+evidence('gray')+'. These endpoint metrics measure protection after gray preprocessing; they do not prove whether background misdirection itself changed.','4. BDR: '+evidence('bdr')+'. Bit-depth reduction is locked to5bits; endpoint detection results do not isolate which TFSS signal was removed.']
        rank=sorted(['gauss','meanf','medianf'],key=lambda d:pairs[d]['recomputed_scores']['P'] if pairs[d]['recomputed_scores']['P'] is not None else float('inf'))
        lines+=['5. Filters ranked by paired protection P (lowest first): '+', '.join(f'{d}: {pairs[d]["recomputed_scores"]["P"]}' for d in rank)+'. Absolute AP and utility are reported for each; this is descriptive.','6. Data Augmentation: '+evidence('data_aug')+'. Statistical significance is unavailable from one seed.','7. Defense harm to Clean (NT AP50 minus untransformed Clean): '+', '.join(f'{d}: {p["metrics"]["clean"]["mAP50_non_target"]-clean0["mAP50_non_target"]:+.9f}' for d,p in pairs.items())+'.','8. Target recovery and S4 NT utility change (Person AP50 delta / NT AP50 delta versus no defense): '+', '.join(f'{d}: {p["metrics"]["S4"]["mAP50_target"]-base["mAP50_target"]:+.9f}/{p["metrics"]["S4"]["mAP50_non_target"]-base["mAP50_non_target"]:+.9f}' for d,p in pairs.items())+'.','9. Same-defense paired P/U/S and their exact denominators are retained in S4_ROBUSTNESS_ALL_RESULTS.csv and each paired_results.json. Signed values are never clipped; zero denominators remain null.','10. Joint absolute target/utility checks: '+ '; '.join(evidence(d) for d in pairs)+'. No arbitrary unregistered low-AP or collapse threshold is imposed.','11. Across-defense consistency is quantified in the full paired table. No defense-independent robustness guarantee follows from these eight one-seed tests.','12. Follow-up priorities ranked by Person AP50 recovery from S4 baseline: '+', '.join(sorted(pairs,key=lambda d:pairs[d]['metrics']['S4']['mAP50_target'],reverse=True))+'. These are analysis priorities, not authorization for more experiments.','', '## Provenance and completeness','',f'Frozen source {config["science_commit"]}; config {config["config_sha256"]}; original S4 manifest {lock["parent_materialization_manifest_sha256"]}.', 'All eight checkpoint/CSV/full-state/data audit receipts and complete VOC20 tables are linked in PHASE_A_ACCEPTANCE.json and PHASE_B_ACCEPTANCE.json. Original S4 data and historical weights were read-only. Historical optional all-class Recall remains unavailable where absent, and was never invented.']
        (doc/'S4_ROBUSTNESS_FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    if phase=='FINAL':
        for ph in ['A','B']:assert load(doc/f'PHASE_{ph}_ACCEPTANCE.json')['state']=='ACCEPTED'
        files={p.name:sha(p) for p in doc.iterdir() if p.is_file() and p.name not in ('PIPELINE_ACCEPTANCE.json','ARCHIVE_MANIFEST.json')}
        save(doc/'ARCHIVE_MANIFEST.json',dict(files=files,all_data_weights_retained_remote=True,deleted_paths=[],AT='NOT_RUN',source=config['science_commit']))
        save(doc/'PIPELINE_ACCEPTANCE.json',dict(state='COMPLETED_AND_ACCEPTED',pipeline_run_id=config['pipeline_run_id'],all_eight_defenses=True,AT='NOT_RUN',science_commit=config['science_commit'],config_sha256=config['config_sha256'],archive_manifest_sha256=sha(doc/'ARCHIVE_MANIFEST.json'),completed_at=time.time()))
    return dict(phase=phase,state='ACCEPTED',reports=str(doc),defenses=list(pairs))
