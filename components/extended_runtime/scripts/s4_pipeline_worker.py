"""Invoke the hash-bound, unchanged S4 computation and its retained audits."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.writing')
    with tmp.open('w') as f:
        json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def bind(config):
    repo=Path(config['science_repo'])
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==config['science_commit']
    for rel,digest in config['science_sha256'].items():assert sha(repo/rel)==digest,rel
    sys.path.insert(0,str(repo))
    spec=importlib.util.spec_from_file_location('frozen_s4_science',repo/'scripts/run_s4_robustness.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    c=load(mod.LOCK);assert sha(mod.LOCK)==config['config_sha256']
    assert c['remote_root']==config['root'] and config['AT_authorized'] is False
    receipt=Path(c['remote_root'])/'receipts/PRERUN_CLOSURE.json'
    assert sha(receipt)==config['review_receipt_sha256']
    r=load(receipt);assert r['pre_run_result']=='pass' and r['pre_run_code_commit']==config['science_commit']
    base=Path(config['output_root'])/'runtime'
    for key,name in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','mpl'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','cache')]:
        p=base/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return mod,c

def historical_valid(c,d):
    h=c['historical_clean'].get(d)
    if not h:return False,'no historical control'
    try:
        a=h['train_args'];expected=dict(c['victim_protocol']['train_args'])
        if d=='data_aug':expected.update(c['defenses']['data_aug']['overrides'])
        for k,v in expected.items():
            if k not in ('data','project','name','exist_ok','patience'):assert a.get(k)==v,(k,a.get(k),v)
        assert sha(Path(h['root'])/'victim/weights/last.pt')==h['last_sha256']
        assert sha(Path(h['root'])/'victim/weights/best.pt')==h['best_sha256']
        assert sha(Path(h['root'])/'last_primary_clean_val_metrics.json')==h['last_metrics_sha256']
        rows=list(csv.DictReader((Path(h['root'])/'victim/results.csv').open()))
        assert len(rows)==200 and [int(float(next(iter(r.values())))) for r in rows]==list(range(1,201))
        return True,'locked200epoch/control/config/checkpoint hashes verified'
    except (OSError,ValueError,AssertionError,KeyError) as e:return False,repr(e)

def effective_config(c,out):
    c=json.loads(json.dumps(c))
    selection=load(out/'selected_arms.json') if (out/'selected_arms.json').exists() else {}
    if selection.get('clean',{}).get('historical') is False:c['historical_clean'].pop(load(out/'pipeline_status.json')['defense'],None)
    return c

def audit(config,mod,c,out,d):
    c=effective_config(c,out)
    payload=json.dumps(dict(output=str(out),config=c,defense=d))
    r=subprocess.run([sys.executable,'-B',str(HERE/'s4_pipeline_frozen_audit.py'),payload],capture_output=True,text=True)
    raw=Path(config['output_root'])/'audit_logs'/f'{out.name}.json'
    dump(raw,dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise RuntimeError('Scientific acceptance failed; raw evidence: '+str(raw))
    answer=next((json.loads(line) for line in reversed(r.stdout.splitlines()) if line.startswith('{') and 'remote_read_only' in line),None)
    assert answer and answer['status']=='passed'
    assert load(out/'paired_results.json')['config_sha256']==config['config_sha256']
    from s4_pipeline_frozen_report import validate_metrics,scores
    pair=load(out/'paired_results.json')
    for arm in ['S4','clean']:validate_metrics(pair['metrics'][arm])
    for k,v in scores(pair['metrics']['clean'],pair['metrics']['S4']).items():
        old=pair['scores'][k]
        assert (v is None and old is None) or (v is not None and old is not None and abs(v-old)<=1e-12)
    return dict(accepted=True,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),output=str(out),**answer)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--action',required=True,choices=['verify','init','historical-check','historical','assemble','audit','finalize','resource','reports']);p.add_argument('--output',type=Path);p.add_argument('--defense');p.add_argument('--phase');p.add_argument('--result',type=Path);a=p.parse_args()
    config=load(a.config);mod,c=bind(config);out=a.output
    if out is not None:assert out.resolve().is_relative_to((Path(c['remote_root'])/'outputs').resolve())
    if a.action=='verify':
        for group,path in config['smoke_runs'].items():
            proof=load(Path(path)/'smoke_summary.json')
            assert proof['state']=='passed' and proof['finite_loss'] and proof['validation_forward_batches']==0
            assert 1<=proof['total_actual_batches']<=100
        assert c['defenses']['bdr']['bits']==5 and c['defenses']['bdr']['status']=='SOURCE_RESOLVED'
        result=dict(passed=True,science_commit=config['science_commit'],config_sha256=config['config_sha256'],BDR_bits=5,prior_actual_training_smokes_verified=True)
    elif a.action=='resource':
        from scripts.run_s4_tfss_pipeline import _resource_snapshot
        result=_resource_snapshot(10.0)
    elif a.action=='init':
        mod.output_setup(out,c);mod.dump(out/'config_snapshot.json',c);mod.parent_audit(c,out,True)
        result=dict(state='running',stage='formal',run_id=out.name,defense=a.defense,source_commit=config['science_commit'],config_sha256=config['config_sha256'],started_at_unix=time.time(),orchestrator_run_id=config['pipeline_run_id'])
        dump(out/'pipeline_status.json',result)
    elif a.action=='historical-check':
        valid,reason=historical_valid(c,a.defense);result=dict(valid=valid,reason=reason)
    elif a.action=='historical':
        valid,reason=historical_valid(c,a.defense);assert valid,reason
        mod.output_setup(out,c);data,_=mod.data_view(c,out,'none','clean',ids=[])
        h=c['historical_clean'][a.defense];weights=Path(h['root'])/'victim/weights'
        m=mod.evaluate(weights/'last.pt',data,out,'last_primary');old=load(Path(h['root'])/'last_primary_clean_val_metrics.json')
        for i,name in enumerate(mod.NAMES):
            assert abs(m['AP50_per_class'][i]-old['AP50_per_class'][name])<=1e-7,'Historical clean AP50 mismatch'
            assert abs(m['Recall_per_class'][i]-old['Recall_per_class'][name])<=1e-7,'Historical clean Recall mismatch'
        mod.evaluate(weights/'best.pt',data,out,'best_secondary')
        result=dict(state='complete',historical=True,source_control=h);dump(out/'pipeline_status.json',result)
    elif a.action=='finalize':
        # Evaluation-only recovery: never calls train or updates a model.
        import torch
        torch.set_num_threads(1)
        rows=list(csv.DictReader((out/'victim_run/victim/results.csv').open()))
        assert len(rows)==200 and [int(float(next(iter(r.values())))) for r in rows]==list(range(1,201))
        weights=out/'victim_run/victim/weights';ck=torch.load(weights/'last.pt',map_location='cpu',weights_only=False)
        embedded=ck['train_results'];key=next(k for k in embedded if k.strip()=='epoch')
        assert len(embedded[key])==200 and int(float(embedded[key][-1]))==200
        assert (out/'victim_run/victim/state_snapshots/epoch_200_full_state.pt').is_file()
        del ck
        if not (out/'victim_completion.json').exists():dump(out/'victim_completion.json',dict(epochs=200,last_sha256=sha(weights/'last.pt'),best_sha256=sha(weights/'best.pt'),recovery='derived from native embedded200CSV; evaluation-only, zero training updates'))
        for role,name in [('last_primary','last.pt'),('best_secondary','best.pt')]:
            dest=out/(role+'_evaluation.json')
            if dest.exists():assert load(dest)['checkpoint_sha256']==sha(weights/name);continue
            evroot=out/'evaluation_recovery'/str(time.time_ns());evroot.mkdir(parents=True)
            mod.evaluate(weights/name,out/'data.yaml',evroot,role)
            assert not dest.exists();dest.write_bytes((evroot/(role+'_evaluation.json')).read_bytes())
        result=dict(state='complete',evaluation_only_recovery=True,training_updates=0);dump(out/'pipeline_status.json',result)
    elif a.action=='assemble':
        from ue_framework.s4_robustness import paired_scores
        result={arm:load(out/arm/'last_primary_evaluation.json') for arm in ['S4','clean']}
        selected=load(out/'selected_arms.json');audits={}
        for arm in ['S4','clean']:
            audits[arm]=dict(reused_clean=True,**c['historical_clean'][a.defense]) if selected[arm].get('historical') else load(out/arm/'data_audit.json')
        pair=dict(defense=a.defense,state='complete',source_commit=config['science_commit'],config_sha256=config['config_sha256'],parent_S4_manifest_sha256=c['parent_materialization_manifest_sha256'],metrics=result,scores=paired_scores(result['clean'],result['S4']),audits=audits,primary='last.pt@200',secondary='best.pt',single_seed_descriptive=True,selected_arms=selected,orchestrator_run_id=config['pipeline_run_id'])
        assert not (out/'paired_results.json').exists();dump(out/'paired_results.json',pair)
        status=load(out/'pipeline_status.json');status.update(state='complete',stage='complete',completed_at_unix=time.time());dump(out/'pipeline_status.json',status);result=dict(complete=True)
    elif a.action=='audit':result=audit(config,mod,c,out,a.defense)
    else:
        from s4_pipeline_reports import generate
        result=generate(config,c,a.phase)
    if a.result:dump(a.result,result)
    print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':main()
