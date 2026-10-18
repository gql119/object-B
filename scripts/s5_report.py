"""S5 identity-safe VOC20 comparison and final acceptance; no model forwards."""
import csv
import hashlib
import json
import math
from pathlib import Path
from scripts.analyze_b2_to_safe_lr_results_20261002 import NAMES,pus,collateral
from scripts.run_s4_tfss_pipeline import comparable_metric_delta


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)


def comparison_tables(metrics):
    for role,value in metrics.items():
        for key in ('AP50','AP75','AP50_95','Recall','Precision'):
            a=value[key+'_per_class']
            if len(a)!=20 or not all(math.isfinite(float(x)) for x in a):raise ValueError('Incomplete metrics '+role)
        value['selectivity']=pus(metrics['Clean'],value);value['collateral']=collateral(metrics['Clean'],value)
    current=metrics['S5'];delta={};missing={}
    for role in ('S0','S1','S2','S3','S4'):
        delta[role],missing[role]=comparable_metric_delta(current,metrics[role])
    aggregate=[]
    for role in ('S0','S1','S2','S3','S4','S5','S5_best_secondary'):
        m=metrics[role];row=dict(Method=role,checkpoint_role=m.get('checkpoint_role'),checkpoint_sha256=m.get('checkpoint_sha256'))
        for key in ('mAP50','mAP75','mAP50_95','Recall','Precision'):
            for part in ('target','non_target','all'):row[key+'_'+part]=m.get(key+'_'+part)
        row.update({k:m['selectivity'][k] for k in ('P','U','S')})
        row.update({k:m['collateral'][k] for k in ('WCD','WCD_class','TCD3')});aggregate.append(row)
    perclass=[]
    for i,name in enumerate(NAMES):
        row=dict(class_id=i,class_name=name)
        for role,value in metrics.items():
            for key in ('AP50','AP75','AP50_95','Recall','Precision'):row[role+'_'+key]=value[key+'_per_class'][i]
        for role in ('S0','S1','S2','S3','S4'):
            for key in ('AP50','AP75','AP50_95','Recall','Precision'):
                row['S5_minus_'+role+'_'+key]=current[key+'_per_class'][i]-metrics[role][key+'_per_class'][i]
        perclass.append(row)
    return dict(state='complete',method='S5=S4+CUP-S4(P3)',primary='last.pt@200',secondary='best.pt',
        metrics=metrics,delta=delta,delta_unavailable_historical_aggregates=missing,
        single_seed_not_significance=True,causal_mechanism_established=False),aggregate,perclass


def cup_tables(out):
    groups={}
    path=out/'noise/cup_classwise_weights.jsonl'
    with path.open() as stream:
        for line in stream:
            r=json.loads(line);key=(r['epoch'],r['cls']);g=groups.setdefault(key,dict(epoch=r['epoch'],cls=r['cls'],
                observations=0,w_sum=0.,w_min=2.,w_max=.5,raw_damage_sum=0.,priority_sum=0.,pi_sum=0.))
            g['observations']+=1;g['w_sum']+=r['w'];g['w_min']=min(g['w_min'],r['w']);g['w_max']=max(g['w_max'],r['w'])
            for dest,source in (('raw_damage_sum','raw_damage'),('priority_sum','priority_before'),('pi_sum','pi')):g[dest]+=r[source]
    rows=[]
    for _,g in sorted(groups.items()):
        n=g['observations'];rows.append(dict(epoch=g['epoch'],cls=g['cls'],class_name=NAMES[g['cls']],
            EOT_class_observations=n,w_mean=g['w_sum']/n,w_min=g['w_min'],w_max=g['w_max'],
            raw_damage_mean=g['raw_damage_sum']/n,normalized_priority_mean=g['priority_sum']/n,instance_pi_mean=g['pi_sum']/n))
    gradient=[]
    with (out/'noise/diagnostics.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('cup_ret_Fourier_norm') not in ('',None):
                gradient.append({k:v for k,v in row.items() if k.startswith('cup_') or k in
                    ('epoch','step','grad_norm_fourier','grad_delta_adaptive','L_ret_base','L_ret_CUP','s4_protection_Fourier_norm')})
    return rows,gradient


def finalize(out,root,commit,lock):
    import torch
    out,root=Path(out),Path(root);reports=out/'reports';reports.mkdir(exist_ok=False)
    completion=json.loads((out/'victim_completion.json').read_text())
    integrity=json.loads((out/'dataset_integrity.json').read_text())
    status=json.loads((out/'noise/status.json').read_text())
    if not (completion['epochs']==completion['rows']==200 and integrity['state']=='passed'
            and integrity['changed_person_gt']==13256 and status['schedule_complete']
            and status['steps']==9144 and status['epochs_completed']==24):raise ValueError('Full serial schedule incomplete')
    if (integrity['train']!=16551 or integrity['val']!=4952 or integrity['target_images']!=6095
            or integrity['non_target_images_unchanged']!=10456
            or not all(integrity[k] for k in ('labels_byte_identical','clean_val_byte_identical',
                'non_target_images_byte_identical','outside_target_zero','per_pixel_frozen_budget_passed','decoded_PNG_audited'))):
        raise ValueError('Full integrity evidence incomplete')
    checkpoint=out/'noise/checkpoints/step_009144_epoch_24_cursor_00000.pt'
    with checkpoint.open('rb') as stream:noise=torch.load(stream,map_location='cpu',weights_only=False)
    if (noise['cursor_state']['step']!=9144 or noise['scheduler']['last_epoch']!=24
            or noise['cup']['history']['steps']!=9144 or noise['cursor_state']['order'] is not None
            or noise['cursor_state']['cursor']!=0):raise ValueError('Noise final state boundary mismatch')
    from ue_framework.methods.s5_cup_history import S5CUPHistory,load_cup_config
    h=S5CUPHistory(load_cup_config(root/lock['s5']['calibration_path']));h.load_state_dict(noise['cup']['history'])
    for values in noise['optimizer']['state'].values():
        if float(values['step'])!=9144 or not torch.isfinite(values['exp_avg']).all() or not torch.isfinite(values['exp_avg_sq']).all():
            raise ValueError('Final Adam state invalid')
    if not all(torch.isfinite(noise['trainer'][k]).all() for k in ('fourier_coeff','freq_score','freq_usage')):
        raise ValueError('Nonfinite Fourier state')
    if sha(noise['cup_journal']['path'])!=noise['cup_journal']['sha256']:raise ValueError('CUP journal integrity mismatch')
    metrics=json.loads((out/'comparison.json').read_text())['metrics']
    for role,name in (('S5','last.pt'),('S5_best_secondary','best.pt')):
        path=out/'victim_run/victim/weights'/name
        if sha(path)!=metrics[role]['checkpoint_sha256']:raise ValueError('Evaluation/checkpoint identity mismatch')
        with path.open('rb') as stream:saved=torch.load(stream,map_location='cpu',weights_only=False)
        result=saved.get('train_results')
        if role=='S5' and (not isinstance(result,dict) or not result or not all(len(v)==200 for v in result.values())):
            raise ValueError('Missing200 embedded primary checkpoint result rows')
    comparison,aggregate,perclass=comparison_tables(metrics)
    write(out/'comparison.json',comparison)
    write(reports/'S5_COMPARISON.json',comparison)
    table(reports/'S5_S0_S3_S4_COMPARISON.csv',aggregate)
    table(reports/'S5_VOC20_PER_CLASS.csv',perclass)
    weights,gradients=cup_tables(out);table(reports/'CUP_CLASSWISE_WEIGHTS.csv',weights);table(reports/'CUP_GRADIENT_AUDIT.csv',gradients)
    write(reports/'S5_NOISE_INTEGRITY.json',integrity)
    write(reports/'S5_CONFIG_LOCK.json',lock)
    m=metrics['S5'];b=metrics['S4'];gate=lock['success_gate']
    criteria=dict(Person_AP50_within_preregistered_limit=m['mAP50_target']<=gate['target_max_literal'],
        NT_AP50_above_literal=m['mAP50_non_target']>gate['NT_AP50_min_literal_exclusive'],
        WCD_below_literal=m['collateral']['WCD']<gate['WCD_max_literal_exclusive'],
        TCD3_below_literal=m['collateral']['TCD3']<gate['TCD3_max_literal_exclusive'],
        NT_AP50_improved_vs_actual_S4=m['mAP50_non_target']>b['mAP50_non_target'],
        WCD_improved_vs_actual_S4=m['collateral']['WCD']<b['collateral']['WCD'],
        TCD3_improved_vs_actual_S4=m['collateral']['TCD3']<b['collateral']['TCD3'],
        NT_AP75_improved_vs_actual_S4=m['mAP75_non_target']>b['mAP75_non_target'])
    unfavorable=[dict(class_name=NAMES[i],AP50_delta=m['AP50_per_class'][i]-b['AP50_per_class'][i])
                 for i in range(20) if i!=14 and m['AP50_per_class'][i]<b['AP50_per_class'][i]]
    required={k:v for k,v in criteria.items() if k!='NT_AP75_improved_vs_actual_S4'}
    verdict='SUCCESS_CANDIDATE_SINGLE_SEED' if all(required.values()) else 'MIXED_OR_REJECTED_BY_PREREGISTERED_CRITERIA'
    acceptance=dict(schema='s5.local-acceptance.v1',accepted=True,run_id=out.name,source_commit=commit,
        config_sha256=sha(root/'docs/S5_CUP_S4/S5_CONFIG_LOCK.json'),noise_checkpoint_sha256=sha(checkpoint),
        victim_completion=completion,noise_steps=9144,noise_epochs=24,CUP_steps=h.steps,
        full_dataset_audit_passed=True,all_class_metrics_complete=True,criteria=criteria,research_verdict=verdict,
        unfavorable_NT_classes=unfavorable,single_seed_not_significance=True,automatic_extra_runs=False)
    body='# S5 final result\n\n'+json.dumps(acceptance,indent=2,allow_nan=False)+'\n\n'
    body+='P3 selected from matched TRAIN32 evidence; S5 changes only detached CUP-S4 class allocation of the original Frozen P3 L_ret. Original TFSS-U/.85, all other S0/S4 losses and weights, Fourier/TSFL/Frozen/Adaptive/SafeBG/budgets/data/native AMP/victim protocol are unchanged. No validation AP tuning or extra seed.\n\n'
    body+='StageB and StageE train proxies do not establish full victim efficacy. Formal last@200 is primary and best is secondary. Differences are descriptive for one seed; no statistical significance, causal interpretation, or proof of layer universality. Negative classes and missing historical optional aggregate Recall/Precision remain explicit.\n\n'
    body+='See complete aggregate/VOC20 CSVs, CUP weights/gradient audit, frozen config and raw SHA-bound artifacts. A positive result only makes S5 a candidate for the main method; this run does not establish robustness over seeds. Stop after this registered run; no automatic tuning or other experiment.\n'
    body+='\n1. Layer: P3, because P5 train-proxy signals were mixed and its opposing protection gradient was larger.\n'
    body+='2. Layer evidence: actual gradients/GT/size/5step audits justify the conservative choice, not a universal superiority claim.\n'
    body+='3. CUP allocation: bounded normalized priorities and actual applied class weights are logged; classwise AP changes are in the VOC20 table and cannot alone prove the proposed mechanism.\n'
    body+='4. Attribution: retention layer stays P3; the designed S4-to-S5 variable is CUP-S4. One seed does not exclude sensitivity or establish causality.\n'
    body+=f"5. Person AP50={m['mAP50_target']}; S4={b['mAP50_target']}; change={m['mAP50_target']-b['mAP50_target']}.\n"
    body+=f"6. NT AP50={m['mAP50_non_target']}; S4={b['mAP50_non_target']}; change={m['mAP50_non_target']-b['mAP50_non_target']}.\n"
    body+=f"7. WCD change={m['collateral']['WCD']-b['collateral']['WCD']}; TCD3 change={m['collateral']['TCD3']-b['collateral']['TCD3']}.\n"
    body+=f'8. Main-method assessment: {verdict}; this remains single-seed descriptive evidence.\n'
    (reports/'S5_FINAL_REPORT.md').write_text(body,encoding='utf-8')
    (reports/'S5_NOISE_TRAINING_REPORT.md').write_text('# S5 noise training\n\n'+json.dumps(dict(status=status,final_checkpoint_sha256=sha(checkpoint),CUP_steps=h.steps,checkpoint_count=len(list((out/'noise/checkpoints').glob('*.pt')))),indent=2)+'\n24epochs/9144updates completed; not a convergence claim. Every50step and epoch checkpoint retained.\n',encoding='utf-8')
    (reports/'S5_VICTIM_TRAINING_REPORT.md').write_text('# S5 victim training\n\n'+json.dumps(completion,indent=2)+'\nFresh random YAML seed0,200continuous epochs,batch36/workers16/AMPtrue/nativeSGD. Last@200 primary, best secondary. Embedded200rows and actual checkpoint SHA verified; stripped epoch=-1 is not an unfinished-run signal.\n',encoding='utf-8')
    for name in ('P3_P5_SOURCE_AUDIT.md','P3_P5_GRADIENT_COMPARISON.csv','P3_P5_NON_TARGET_RETENTION.csv',
                 'P3_P5_5STEP_RESULTS.md','P3_P5_LAYER_DECISION.md','CUP_S4_DESIGN.md','CUP_S4_ADAPTATION_DIFF.md'):
        source=root/'docs/S5_CUP_S4'/name
        if not source.exists():raise ValueError('Missing final required evidence '+name)
        (reports/name).write_bytes(source.read_bytes())
    smoke_path=out.parents[1]/'outputs'/lock['s5']['smoke_run_id']/'smoke_summary.json'
    smoke=json.loads(smoke_path.read_text())
    if smoke['state']!='complete' or smoke['source_commit']!=commit:raise ValueError('Smoke attribution changed')
    (reports/'S5_5STEP_SMOKE.md').write_text('# Current-source S5 production smoke\n\n'+json.dumps(smoke,indent=2,allow_nan=False)+'\n\nNo official metrics; bounded TRAIN32 observations, not formal efficacy.\n',encoding='utf-8')
    (reports/'S5_PARENT_EQUIVALENCE.md').write_text('# Actual S4/off and neutral-CUP equivalence\n\n'+json.dumps(dict(source_commit=commit,smoke_path=str(smoke_path),smoke_sha256=sha(smoke_path),max_difference=smoke['original_S4_vs_P3_noop_max_difference'],tolerance=1e-7,resume=smoke['resume'],native_AMP=smoke['native_AMP']),indent=2)+'\n',encoding='utf-8')
    write(reports/'LOCAL_ACCEPTANCE.json',acceptance)
    return acceptance
