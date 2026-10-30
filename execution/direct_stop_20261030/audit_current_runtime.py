"""Small current scientific-source audit; no model/GPU/data census or cleanup."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);args=a.parse_args()
    p=json.loads((ROOT/'robustness_s4_at/configs/PROTOCOL.json').read_text(encoding='utf8'))
    files=sorted((ROOT/'robustness_s4_at').glob('*.py'))+[ROOT/'robustness_s4_at/configs/PROTOCOL.json']
    records=[];different=[]
    for path in files:
        rel=path.relative_to(ROOT).as_posix();actual=digest(path)
        current=subprocess.check_output(['git','show',args.source+':'+rel],cwd=ROOT)
        if hashlib.sha256(current).hexdigest()!=actual:raise ValueError('Current source not committed: '+rel)
        previous=subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT)
        exact=hashlib.sha256(previous).hexdigest()==actual
        if not exact:different.append(rel)
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf8'))
        records.append(dict(path=rel,sha256=actual,byte_exact_original_accepted_runtime=exact))
    permitted={'robustness_s4_at/common.py','robustness_s4_at/trainer.py','robustness_s4_at/worker.py','robustness_s4_at/configs/PROTOCOL.json'}
    if set(different)-permitted:raise ValueError('Unselected scientific computation changed: '+str(different))
    if p['train_args']['batch']==36 and different:raise ValueError('OriginalBatch36 fallback must retain the accepted scientific package byte-exact')
    original=[]
    for rel,expected in p['original_sources'].items():
        actual=digest(ROOT/rel)
        if actual!=expected:raise ValueError('Original S4 generation source changed: '+rel)
        original.append(dict(path=rel,sha256=actual))
    if p['pgd']!={'4':dict(epsilon_pixels=4,step_pixels=1,steps=10,random_start=True),'8':dict(epsilon_pixels=8,step_pixels=2,steps=10,random_start=True)}:raise ValueError('PGD protocol drift')
    if (p['target_id'],p['nc'],p['train_count'],p['val_count'])!=(14,20,16551,4952):raise ValueError('Data definition changed')
    t=p['train_args'];batch=t['batch'];accumulate=max(round(t['nbs']/batch),1);decay=t['weight_decay']*batch*accumulate/t['nbs']
    if batch not in (36,48,64,72) or decay!=0.0005625000000000001:raise ValueError('Declared native batch/actual decay mismatch')
    v=dict(state='PASS_COMMITTED_SOURCE_AND_CPU_REUSE_NOT_GPU',source_commit=args.source,prior_accepted_source=BASE,
        scientific_files=records,intentionally_changed_scientific_files=different,OriginalS4_sources=original,
        source_only_not_models=True,models=0,GPU_work=False,CPU_native_evidence_reused='accepted CPU contracts and native loops; unchanged computation only',
        current_GPU_evidence_still_required=True,train_args=t,native_actual_weight_decay=decay,
        nominal_postwarmup_accumulate=accumulate,PGD=p['pgd'],formal_ready=False)
    with (HERE/'CURRENT_RUNTIME_CPU_REUSE.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
    print(json.dumps(dict(state=v['state'],source=args.source,scientific_files=len(records),original_sources=len(original),changed=different,batch=batch,actual_decay=decay)))
