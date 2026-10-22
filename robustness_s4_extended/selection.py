"""Conservative post-acceptance selection. No training or projection code.
Rule is frozen before Full metrics exist; a failed Full never implies original wins.
One immutable Decision JSON supplies every later Clean/Poison/smoke/OP identity.
"""
import copy,csv,datetime,json,math
from pathlib import Path
from .common import ROOT,load,sha,atomic,commit
from .metrics import paired

class SelectionBlocked(ValueError):pass

def validate_rule(rule):
    if rule.get('decision_rule_version')!='conservative_pareto_S_20261022_v1' or rule.get('S_improvement_tolerance')!=1e-9 or rule.get('other_guard_tolerance')!=0:
        raise SelectionBlocked('Selection rule drift; no post-result threshold changes')
    return rule

def compare_metrics(clean,s4,full,rule, *, full_validated, protocols_equal):
    validate_rule(rule)
    if not full_validated or not protocols_equal:raise SelectionBlocked('BLOCKED_FULL_VALIDATION')
    a,b=paired(clean,s4),paired(clean,full)
    if any(q.get(k) is None for q in (a,b) for k in ['S','WCD','P','U']):raise SelectionBlocked('Undefined selection score')
    def row(m,s):
        nt=[i for i in range(20) if i!=14]
        return dict(Person_AP50=m['AP50_per_class'][14],NT_mAP50=sum(m['AP50_per_class'][i] for i in nt)/19,
            NT_mAP75=sum(m['AP75_per_class'][i] for i in nt)/19,S=s['S'],P=s['P'],U=s['U'],WCD=s['WCD'],TCD3=s['TCD3'],
            NT_AP50={str(i):m['AP50_per_class'][i] for i in nt})
    left,right=row(s4,a),row(full,b)
    checks=dict(S_improves_over_tolerance=right['S']-left['S']>rule['S_improvement_tolerance'],
        Person_AP50_not_higher=right['Person_AP50']<=left['Person_AP50'],
        NT_mAP50_not_lower=right['NT_mAP50']>=left['NT_mAP50'],
        WCD_not_higher=right['WCD']<=left['WCD'],identical_evaluation_protocol=protocols_equal)
    return dict(selected_method='S4_TFSS_FULL' if all(checks.values()) else 'S4_ORIGINAL',
        comparison_metrics=dict(S4_ORIGINAL=left,S4_TFSS_FULL=right),checks=checks,
        selection_reason='All frozen conservative advantage conditions hold' if all(checks.values()) else 'Full did not satisfy every frozen advantage condition; descriptive conservative default',
        single_seed_descriptive=True,statistical_significance_claim=False)

def descriptor(c,method):
    if method=='S4_ORIGINAL':
        b=c['baseline']
        return dict(dataset=b['poisoned_dataset_path'],manifest=b['parent_output']+'/materialization_manifest.json',
            manifest_sha256=b['parent_materialization_manifest_sha256'],integrity_sha256=b['parent_dataset_integrity_sha256'],
            parent_output=b['parent_output'],generation_run_id=b['parent_noise_run_id'],noise_sha256=b['parent_noise_checkpoint_sha256'])
    if method=='S4_TFSS_FULL':
        p=c['predecessor'];r=p['recovery'];b=r['parent_files']
        return dict(dataset=r['generation_output']+'/dataset',manifest=r['generation_output']+'/materialization_manifest.json',
            manifest_sha256=b['materialization_manifest.json'],integrity_sha256=b['dataset_integrity.json'],
            parent_output=r['generation_output'],generation_run_id=r['generation_run_id'],
            noise_sha256=b['noise/checkpoints/step_009144_epoch_24_cursor_00000.pt'])
    raise SelectionBlocked('Unknown method; never guess a dataset')

def decision_path(c):return Path(c['output_root'])/'S4_FULL_DECISION.json'

def load_decision(c, *, check_files=True):
    p=decision_path(c);d=load(p)
    origin=c['selection'].get('locked_decision_reuse')
    expected=(origin['batch_id'],origin['source_commit'],origin['config_sha256']) if origin else (c['pipeline_run_id'],commit(),sha(ROOT/'robustness_s4_extended/configs/QUEUE.json'))
    if origin and sha(p)!=origin['decision_sha256']:raise SelectionBlocked('Recovered immutable Decision bytes changed')
    if (d.get('batch_id'),d.get('selection_source_commit'),d.get('selection_config_sha256'),d.get('decision_rule_sha256'),d.get('validation_status'))!=(*expected,c['selection']['rule_sha256'],'VALIDATED_AND_LOCKED'):
        raise SelectionBlocked('Selected batch/source/rule/config differs')
    x=descriptor(c,d['selected_method'])
    expected=dict(selected_poison_dataset=x['dataset'],selected_dataset_manifest=x['manifest'],
        selected_dataset_sha256=x['manifest_sha256'],selected_integrity_sha256=x['integrity_sha256'])
    if any(d.get(k)!=v for k,v in expected.items()):raise SelectionBlocked('Selected data changed or mixed')
    if check_files and (sha(x['manifest'])!=x['manifest_sha256'] or sha(Path(x['parent_output'])/'dataset_integrity.json')!=x['integrity_sha256']):
        raise SelectionBlocked('Selected manifest/integrity drift')
    return d

def runtime_config(c):
    d=load_decision(c);x=descriptor(c,d['selected_method']);v=copy.deepcopy(c['baseline'])
    v.update(remote_root=c['root'],poisoned_dataset_path=d['selected_poison_dataset'],
        parent_output=x['parent_output'],parent_noise_run_id=x['generation_run_id'],
        parent_noise_checkpoint_sha256=x['noise_sha256'],parent_dataset_integrity_sha256=x['integrity_sha256'],
        parent_materialization_manifest_sha256=x['manifest_sha256'])
    return v

def decision_identity(c):
    d=load_decision(c)
    return dict(selected_method=d['selected_method'],selected_dataset_manifest=d['selected_dataset_manifest'],
        selected_dataset_sha256=d['selected_dataset_sha256'],decision_sha256=sha(decision_path(c)),batch_id=d['batch_id'])

def select_and_lock(c):
    path=decision_path(c)
    if path.exists():return load_decision(c)
    root=Path(c['output_root']);proof=load(root/'PREDECESSOR_ACCEPTANCE.json')
    if proof.get('state')!='COMPLETED_AND_ACCEPTED' or proof.get('run_id')!=c['predecessor']['run_id']:
        raise SelectionBlocked('Full acceptance required before comparison')
    state=load(Path(c['predecessor']['output'])/'pipeline_status.json')
    if state.get('state')!='complete' or sha(Path(c['predecessor']['output'])/'pipeline_status.json')!=proof['native_state_sha256']:
        raise SelectionBlocked('Full actual acceptance state changed')
    origin=c['selection'].get('locked_decision_reuse')
    if origin:
        previous=Path(origin['output_root'])
        data=(previous/'S4_FULL_DECISION.json').read_bytes()
        if sha(previous/'S4_FULL_DECISION.json')!=origin['decision_sha256'] or sha(root/'PREDECESSOR_ACCEPTANCE.json')!=origin['full_acceptance_sha256']:
            raise SelectionBlocked('Recovered decision/predecessor evidence changed')
        # Exact bytes, including the original batch/source/config attribution.
        # Recovery cannot reselect a method or rewrite a decision already used.
        with path.open('xb') as stream:stream.write(data)
        for name in ['S4_FULL_COMPARISON.csv','S4_FULL_COMPARISON.md']:
            with (root/name).open('xb') as stream:stream.write((previous/name).read_bytes())
        return load_decision(c)
    queue=c['selection']['original_robustness'];old=load(queue['pipeline_state_path'])
    if old.get('pipeline_run_id')!=queue['pipeline_id'] or old.get('state')!='COMPLETED_AND_ACCEPTED' or old.get('active_job') is not None or not all(old['stages'][s]['state']=='ACCEPTED' for s in queue['required_stages']):
        raise SelectionBlocked('Original eight-defense pipeline not accepted')
    rulepath=ROOT/c['selection']['rule_path'];rule=validate_rule(load(rulepath))
    if sha(rulepath)!=c['selection']['rule_sha256']:raise SelectionBlocked('Frozen rule bytes changed')
    ref=load(ROOT/'robustness_s4_extended/configs/FULL_REFERENCE_LOCK.json')
    baseline=ROOT/ref['baseline']['comparison_path']
    if sha(baseline)!=ref['baseline']['comparison_sha256']:raise SelectionBlocked('Original S4 results changed')
    b=load(baseline);f=Path(c['predecessor']['output'])
    if b['primary']!='last.pt@200' or b['training']['epochs']!=200 or b['training']['results_rows']!=200:
        raise SelectionBlocked('S4 primary full200 identity missing')
    original=Path(b['training']['args']['project'])/b['training']['args']['name']
    if sha(original/'weights/last.pt')!=ref['baseline']['last_sha256']:
        raise SelectionBlocked('Original S4 actual last checkpoint changed')
    history=list(csv.DictReader((original/'results.csv').open()))
    if len(history)!=200 or [int(float(next(iter(row.values())))) for row in history]!=list(range(1,201)):
        raise SelectionBlocked('Original S4 actual epoch200 CSV missing')
    parent=Path(ref['baseline']['output'])
    for rel,key in [('materialization_manifest.json','manifest_sha256'),('dataset_integrity.json','integrity_sha256'),(ref['baseline']['noise_checkpoint_relative_path'],'noise_sha256')]:
        if sha(parent/rel)!=ref['baseline'][key]:raise SelectionBlocked('Original S4 source artifact changed: '+rel)
    spec=load(f/'victim_run_spec.json')
    def normalized(a):return {k:v for k,v in a.items() if k not in ['data','project']}
    protocols=(normalized(spec['train_args'])==normalized(b['training']['args']) and spec['initialization']=='random_yaml' and spec['init_yaml_sha256']==ref['victim_protocol']['init_yaml_sha256'])
    if sha(f/'config_lock_snapshot.json')!=c['predecessor']['config_sha256']:raise SelectionBlocked('Full lock differs')
    full=load(f/'S4_TFSS_FULL_OVERLAP_evaluation.json');s4=b['metrics']['S4']
    if full['checkpoint_sha256']!=load(f/'victim_completion.json')['last_sha256'] or full['checkpoint_role']!='last.pt@200':
        raise SelectionBlocked('Full checkpoint role/identity mismatch')
    if s4.get('checkpoint_sha256')!=ref['baseline']['last_sha256']:raise SelectionBlocked('Original S4 last checkpoint mismatch')
    result=compare_metrics(b['metrics']['Clean'],s4,full,rule,full_validated=True,protocols_equal=protocols)
    x=descriptor(c,result['selected_method'])
    d=dict(**result,batch_id=c['pipeline_run_id'],selected_poison_dataset=x['dataset'],
        selected_dataset_manifest=x['manifest'],selected_dataset_sha256=x['manifest_sha256'],
        selected_dataset_sha256_definition='SHA256 of complete per-image/label materialization manifest, independently verified',
        selected_integrity_sha256=x['integrity_sha256'],decision_rule_version=rule['decision_rule_version'],
        decision_rule_sha256=sha(rulepath),decision_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validation_status='VALIDATED_AND_LOCKED',selection_source_commit=commit(),
        selection_config_sha256=sha(ROOT/'robustness_s4_extended/configs/QUEUE.json'),
        full_acceptance_sha256=sha(root/'PREDECESSOR_ACCEPTANCE.json'),
        original_results_sha256=sha(baseline),Full_last_sha256=full['checkpoint_sha256'],
        original_robustness_state_sha256=sha(queue['pipeline_state_path']),selected_batch_can_change=False)
    # O_EXCL creates an immutable decision, never overwrites one used by a job.
    with path.open('x',encoding='utf-8') as stream:json.dump(d,stream,indent=2,allow_nan=False);stream.write('\n')
    rows=[dict(method=k,**{a:v for a,v in m.items() if a!='NT_AP50'}) for k,m in result['comparison_metrics'].items()]
    with (root/'S4_FULL_COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# S4 / Full automatic comparison','',f"Selected Method: {d['selected_method']}",'',d['selection_reason'],
        '',f"Frozen S improvement tolerance: {rule['S_improvement_tolerance']}; other guards use exact directional comparisons.",
        '',json.dumps(result['comparison_metrics'],indent=2),'','Single-seed descriptive scheduling rule; no significance or causal mechanism claim.']
    (root/'S4_FULL_COMPARISON.md').write_text('\n'.join(text),encoding='utf-8')
    return load_decision(c)

