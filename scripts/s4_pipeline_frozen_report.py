"""Derive tables only from accepted, completed paired runs. No training or GPU."""
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/S4_ROBUSTNESS_20261019'
SMALL=ROOT.parent/'s4_robustness_small_20261019'
ORDER=['none','gauss','jpeg50','jpeg30','meanf','medianf','gray','bdr','at','data_aug']
LABEL=['Clean','Gauss','JPEG50','JPEG30','MeanF','MedianF','Gray','BDR','AT','Data Augmentation']
NAMES=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
PHASES={'A':['jpeg50','jpeg30','gray','bdr'],'B':['gauss','meanf','medianf','data_aug']}
METRICS=['mAP50_target','mAP50_non_target','mAP75_target','mAP50_95_target','Recall_target','Precision_target','mAP75_non_target','mAP50_95_non_target','Recall_non_target','Precision_non_target','mAP50_all','mAP75_all','mAP50_95_all','Recall_all','Precision_all']
PERCLASS=['AP50','AP75','AP50_95','Recall','Precision']

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writejson(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writecsv(p,rows,fields):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def scores(a,b):
    # Independent reconstruction rather than reuse the producing function.
    p=None if a['mAP50_target']==0 else 1-b['mAP50_target']/a['mAP50_target']
    u=None if a['mAP50_non_target']==0 else b['mAP50_non_target']/a['mAP50_non_target']
    s=None if p is None or u is None or p+u==0 else 2*p*u/(p+u)
    losses=[None if x==0 else (x-y)/x for i,(x,y) in enumerate(zip(a['AP50_per_class'],b['AP50_per_class'])) if i!=14]
    ordered=sorted([v for v in losses if v is not None],reverse=True)
    return dict(P=p,U=u,S=s,WCD=ordered[0] if len(ordered)==19 else None,TCD3=sum(ordered[:3])/3 if len(ordered)==19 else None)

def validate_metrics(m):
    for key in PERCLASS:
        v=m[key+'_per_class'];assert len(v)==20
        assert all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in v)
    for key,source in [('mAP50','AP50'),('mAP75','AP75'),('mAP50_95','AP50_95'),('Recall','Recall'),('Precision','Precision')]:
        v=m[source+'_per_class'];nt=[x for i,x in enumerate(v) if i!=14]
        for suffix,expected in [('target',v[14]),('non_target',sum(nt)/19),('all',sum(v)/20)]:
            if key+'_'+suffix in m:assert math.isclose(m[key+'_'+suffix],expected,abs_tol=1e-12)

def accept(defense,lock,lockhash):
    rid=f's4_robustness_{defense}_paired_seed0_20261019_v1';p=SMALL/rid
    accepted=ROOT/'execution/acceptance'/f'{rid}.json'
    if not accepted.is_file():
        state='IN_PROGRESS' if (ROOT/'execution/state'/rid/'run_ref.json').is_file() else 'NOT_RUN'
        return None,dict(run_id=rid,state=state,accepted=False)
    receipt=load(accepted);assert receipt['accepted'] is True
    pair=load(p/'paired_results.json');status=load(p/'pipeline_status.json')
    assert status['state']=='complete' and pair['state']=='complete' and pair['defense']==defense
    assert receipt['paired_results_sha256']==sha(p/'paired_results.json')
    assert pair['config_sha256']==lockhash
    assert pair['parent_S4_manifest_sha256']==lock['parent_materialization_manifest_sha256']
    assert pair['source_commit']==receipt['source_commit']
    for arm in ['S4','clean']:
        validate_metrics(pair['metrics'][arm])
        ev=load(p/arm/'last_primary_evaluation.json')
        assert ev==pair['metrics'][arm]
        assert ev['checkpoint_role']=='last_primary'
        reused=arm=='clean' and defense in lock['historical_clean']
        if reused:assert ev['checkpoint_sha256']==lock['historical_clean'][defense]['last_sha256']
        else:
            audit=load(p/arm/'data_audit.json');completion=load(p/arm/'victim_completion.json')
            assert audit['count']==16551 and audit['labels_byte_identical'] and audit['geometry_unchanged']
            assert audit['source_dataset']==lock['clean_root' if arm=='clean' else 'poisoned_dataset_path']
            rows=list(csv.DictReader((p/arm/'victim_run/victim/results.csv').open()))
            assert len(rows)==200 and [int(float(next(iter(r.values())))) for r in rows]==list(range(1,201))
            assert completion['epochs']==200 and completion['last_sha256']==ev['checkpoint_sha256']
    pair['recomputed_scores']=scores(pair['metrics']['clean'],pair['metrics']['S4'])
    for k,v in pair['recomputed_scores'].items():
        old=pair['scores'][k];assert (v is None and old is None) or (v is not None and old is not None and math.isclose(v,old,abs_tol=1e-12))
    return pair,dict(run_id=rid,state='COMPLETE_ACCEPTED',accepted=True,paired_results_sha256=sha(p/'paired_results.json'))

def main():
    lock=load(DOC/'S4_ROBUSTNESS_LOCK.json');lockhash=sha(DOC/'S4_ROBUSTNESS_LOCK.json')
    baseline=load(ROOT/'docs/S4_AMP_RECOVERY_20261018/S4_COMPARISON.json')['metrics']
    data={'none':{'metrics':{'clean':baseline['Clean'],'S4':baseline['S4']},'recomputed_scores':scores(baseline['Clean'],baseline['S4'])}}
    states={};allrows=[];extended=[];perclass=[];manifests=[]
    for defense in PHASES['A']+PHASES['B']:
        pair,state=accept(defense,lock,lockhash);states[defense]=state
        if pair:data[defense]=pair
    for defense in ORDER:
        pair=data.get(defense);status='COMPLETE_ARCHIVED' if defense=='none' else states.get(defense,{'state':'NOT_RUN'})['state']
        for arm in ['clean','S4']:
            m=pair['metrics'][arm] if pair else {}
            row=dict(defense=defense,arm=arm,status=status,run_id=states.get(defense,{}).get('run_id','archived_S4_or_Clean' if defense=='none' else 'NOT_RUN'),checkpoint_sha256=m.get('checkpoint_sha256','NOT_RUN'),seed=0,primary='last.pt@200',source_commit=pair.get('source_commit','archived provenance in S4_COMPARISON.json') if pair else 'NOT_RUN',config_sha256=pair.get('config_sha256','archived S4 lock') if pair else lockhash,parent_S4_manifest_sha256=lock['parent_materialization_manifest_sha256'],defense_parameters=json.dumps(lock['defenses'].get(defense,{}),sort_keys=True))
            row.update({k:m.get(k,None) if pair else status for k in METRICS})
            sc=pair['recomputed_scores'] if pair and arm=='S4' else {}
            row.update({k:sc.get(k,None) if pair or arm=='clean' else status for k in ['P','U','S','WCD','TCD3']})
            allrows.append(row)
            for metric in METRICS:
                base=baseline['S4' if arm=='S4' else 'Clean'].get(metric)
                value=m.get(metric)
                extended.append(dict(defense=defense,arm=arm,status=status,metric=metric,value=value if pair else status,untransformed_baseline=base,defense_minus_baseline=None if value is None or base is None else value-base))
            if pair:
                for i,name in enumerate(NAMES):
                    perclass.append(dict(defense=defense,arm=arm,status=status,class_id=i,class_name=name,**{k:m[k+'_per_class'][i] for k in PERCLASS}))
            if pair and defense!='none':
                audit=pair['audits'][arm]
                manifests.append(dict(defense=defense,arm=arm,run_id=states[defense]['run_id'],manifest_sha256=audit.get('manifest_sha256','historical_clean_reuse'),parent_S4_manifest_sha256=lock['parent_materialization_manifest_sha256'],checkpoint_sha256=m['checkpoint_sha256']))
    fields=list(allrows[0]);writecsv(DOC/'S4_ROBUSTNESS_ALL_RESULTS.csv',allrows,fields)
    writecsv(DOC/'S4_ROBUSTNESS_EXTENDED_METRICS.csv',extended,list(extended[0]))
    writecsv(DOC/'S4_DEFENSE_DATA_MANIFEST.csv',manifests,['defense','arm','run_id','manifest_sha256','parent_S4_manifest_sha256','checkpoint_sha256'])
    table=['# S4 robustness main table','', 'All numbers are fractions; last@200 primary. AT NOT_RUN. Clean S is not applicable.','', '| Arm / Metric | '+' | '.join(LABEL)+' |','|---|'+'---:|'*len(ORDER)]
    for arm in ['clean','S4']:
        for title,key in [('Person AP50','mAP50_target'),('Non-target mAP50','mAP50_non_target'),('Selectivity S','S')]:
            cells=[]
            for defense in ORDER:
                pair=data.get(defense)
                if key=='S' and arm=='clean':value='—'
                elif not pair:value=states.get(defense,{'state':'NOT_RUN'})['state']
                else:
                    v=pair['recomputed_scores'].get('S') if key=='S' else pair['metrics'][arm][key]
                    value='UNDEFINED' if v is None else format(v,'.9f')
                cells.append(value)
            table.append('| '+arm+' '+title+' | '+' | '.join(cells)+' |')
    (DOC/'S4_ROBUSTNESS_MAIN_TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    for phase,defenses in PHASES.items():
        rows=[r for r in allrows if r['defense'] in defenses]
        writecsv(DOC/f'PHASE_{phase}_ROBUSTNESS_RESULTS.csv',rows,fields)
        writecsv(DOC/f'PHASE_{phase}_PER_CLASS_RESULTS.csv',[r for r in perclass if r['defense'] in defenses],['defense','arm','status','class_id','class_name',*PERCLASS])
        complete=all(states[d]['accepted'] for d in defenses)
        writejson(DOC/f'PHASE_{phase}_TRAINING_STATUS.json',dict(phase=phase,state='COMPLETE_ACCEPTED' if complete else 'INCOMPLETE',defenses={d:states[d] for d in defenses},generated_utc=datetime.now(timezone.utc).isoformat()))
        lines=[f'# Phase {phase} report','',f'Status: {"COMPLETE_ACCEPTED" if complete else "INCOMPLETE"}. Only full accepted results are included.','']
        for d in defenses:
            if d not in data:lines.append(f'- {d}: {states[d]["state"]}; no efficacy conclusion.');continue
            p=data[d];a=p['metrics']['clean'];b=p['metrics']['S4'];s=p['recomputed_scores']
            lines.append(f'- {d}: Clean/S4 Person AP50={a["mAP50_target"]:.9f}/{b["mAP50_target"]:.9f}; NT mAP50={a["mAP50_non_target"]:.9f}/{b["mAP50_non_target"]:.9f}; P/U/S={s["P"]}/{s["U"]}/{s["S"]}.')
        lines+=['','Target AP/Recall changes are descriptive, paired against the same defense Clean. Single seed provides no significance or causal proof. Strong defenses may harm Clean detection; high S alone is not overall superiority.']
        (DOC/f'PHASE_{phase}_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    done=all(s['accepted'] for s in states.values())
    questions=['JPEG50 protection','JPEG30 versus JPEG50','Gray and target protection','BDR and protection','Gaussian/Mean/Median comparison','Data augmentation','NT degradation','Target recovery versus NT loss','Paired P/U/S','Low Person AP with NT collapse','Consistency across defenses','Further analysis priorities']
    final=['# S4 robustness final report','',f'Phase A/B evidence status: {"COMPLETE_ACCEPTED; final research interpretation still required" if done else "INCOMPLETE"}. AT remains NOT_RUN; no AT training authorized.','', 'Results: S4_ROBUSTNESS_MAIN_TABLE.md, ALL_RESULTS.csv, EXTENDED_METRICS.csv and phase per-class CSVs.','', 'This generated draft is not final research acceptance. The twelve questions require explicit evidence-based answers before publication and stopping the monitor.','']
    for i,q in enumerate(questions,1):final.append(f'{i}. {q}: '+('OPEN for interpretation using the accepted paired absolute AP/Recall and P/U/S; no multi-seed significance claim.' if done else 'OPEN until the required paired 200-epoch results pass acceptance.'))
    (DOC/'S4_ROBUSTNESS_FINAL_REPORT.md').write_text('\n'.join(final)+'\n',encoding='utf-8')
    print(json.dumps({'phase_A_accepted':all(states[d]['accepted'] for d in PHASES['A']),'phase_B_accepted':all(states[d]['accepted'] for d in PHASES['B']),'states':states}))

if __name__=='__main__':main()
