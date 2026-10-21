"""Read-only server verification and scoped final-report preparation. No GPU or training."""
from pathlib import Path
import csv, datetime, hashlib, json, math, re, sys
from remote_run_control.profiles import ProfileStore
from remote_run_control.transport import transport_for

W=Path(__file__).resolve().parents[1]
SMALL=W.parent/'s4_robustness_small_20261019/s4_robustness_server_pipeline_seed0_20261020_recovery01'
DOC=SMALL/'reports'
DEST=W/'docs/S4_ROBUSTNESS_FINAL_20261022'
cfg=json.loads((W/'docs/S4_ROBUSTNESS_PIPELINE_20261020/PIPELINE_RECOVERY01_CONFIG.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

REMOTE=r'''
from pathlib import Path
import csv,getpass,hashlib,json,sys
p=json.loads(sys.argv[1]);cfg=p['config'];root=Path(cfg['output_root']);state=json.loads((root/'pipeline_state.json').read_text())
assert getpass.getuser()=='guoqinglong'
assert state['pipeline_run_id']==cfg['pipeline_run_id'] and state['state']=='COMPLETED_AND_ACCEPTED' and state['active_job'] is None
assert all(state['stages'][x]['state']=='ACCEPTED' for x in cfg['order'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
lock=load(Path(cfg['science_repo'])/'docs/S4_ROBUSTNESS_20261019/S4_ROBUSTNESS_LOCK.json')
pairs={}
for d in ['jpeg50','jpeg30','gray','bdr','gauss','meanf','medianf','data_aug']:
 out=Path(cfg['root'])/'outputs'/cfg['bootstrap_runs'][d] if d in cfg['bootstrap_runs'] else Path(cfg['stages'][d]['output'])
 pair=load(out/'paired_results.json');checks={}
 for arm in ['clean','S4']:
  a=out/arm;ev=load(a/'last_primary_evaluation.json');best=load(a/'best_secondary_evaluation.json')
  hist=lock['historical_clean'].get(d) if arm=='clean' else None
  vr=Path(hist['root'])/'victim' if hist else a/'victim_run/victim'
  rows=list(csv.DictReader((vr/'results.csv').open()))
  assert len(rows)==200 and [int(float(next(iter(r.values())))) for r in rows]==list(range(1,201))
  lastsha=sha(vr/'weights/last.pt');bestsha=sha(vr/'weights/best.pt')
  assert lastsha==ev['checkpoint_sha256']==pair['metrics'][arm]['checkpoint_sha256'] and bestsha==best['checkpoint_sha256']
  assert pair['metrics'][arm]==ev
  checks[arm]={'csv_rows':200,'last_sha256':lastsha,'best_sha256':bestsha,'historical_clean_reuse':bool(hist)}
  if hist:assert lastsha==hist['last_sha256'] and bestsha==hist['best_sha256']
  else:
   au=load(a/'data_audit.json');comp=load(a/'victim_completion.json');assert au['count']==16551 and au['labels_byte_identical'] and au['geometry_unchanged']
   assert comp['epochs']==200 and comp['last_sha256']==lastsha
   assert sha(a/'data_manifest.json')==au['manifest_sha256']
   checks[arm]['dataset_manifest_sha256']=au['manifest_sha256']
 pairs[d]={'pair':pair,'paired_results_sha256':sha(out/'paired_results.json'),'checks':checks}
repo=Path(p['binding_repo']);components={}
for base in [root,repo]:
 q=base/'source_components/report_binding_update.json'
 if q.is_file():components[str(q)]={'sha256':sha(q),'content':load(q)}
for rel in ['scripts/s4_pipeline_reports.py','scripts/s4_pipeline_frozen_report.py']:
 components[rel]={'sha256':sha(repo/rel)}
assert components
print(json.dumps({'user':getpass.getuser(),'state_sha256':sha(root/'pipeline_state.json'),'pairs':pairs,'source_components':components,'remote_read_only':True,'GPU_forwards':0},allow_nan=False))
'''

def main():
 check=load(W/'execution/s4_robustness_server_pipeline_seed0_20261020_recovery01_CHECK_LATEST.json')
 assert check['wait']['status']['state']=='completed' and check['health']['healthy'] and check['health']['phase']=='completion'
 receipt=load(DOC/'PIPELINE_ACCEPTANCE.json');archive=load(DOC/'ARCHIVE_MANIFEST.json')
 assert receipt['state']=='COMPLETED_AND_ACCEPTED' and receipt['archive_manifest_sha256']==sha(DOC/'ARCHIVE_MANIFEST.json')
 for n,h in archive['files'].items():assert sha(DOC/n)==h,(n,'archive hash')
 phases={}
 for phase in ['A','B']:
  p=load(DOC/f'PHASE_{phase}_ACCEPTANCE.json');assert p['state']=='ACCEPTED'
  for key,name in [('report_sha256',f'PHASE_{phase}_FINAL_REPORT.md'),('results_sha256',f'PHASE_{phase}_RESULTS.csv'),('per_class_sha256',f'PHASE_{phase}_PER_CLASS.csv')]:assert p[key]==sha(DOC/name)
  phases.update(p['defenses'])
 transport=transport_for(ProfileStore(W/'execution/profiles.json').load('school-s1'))
 spec=load(W/'execution/s4_robustness_server_pipeline_seed0_20261020_recovery01.json')
 r=transport.run([cfg['python'],'-B','-',json.dumps({'config':cfg,'binding_repo':spec['remote']['repo_root']})],input_data=REMOTE.encode())
 raw={'returncode':r.returncode,'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace')}
 (W/'execution/FINAL_RESULTS_REMOTE_RAW_20261022.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
 assert r.returncode==0,raw['stderr']
 evidence=json.loads(raw['stdout']);assert len(evidence['pairs'])==8
 summaries=[]
 for d,item in evidence['pairs'].items():
  p=item['pair'];assert item['paired_results_sha256']==phases[d]['paired_results_sha256']
  assert p['source_commit']==cfg['science_commit'] and p['config_sha256']==cfg['config_sha256']
  assert p['parent_S4_manifest_sha256']=='194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36'
  for arm,m in p['metrics'].items():
   for key,arr in [('mAP50','AP50'),('mAP75','AP75'),('mAP50_95','AP50_95'),('Recall','Recall'),('Precision','Precision')]:
    vals=m[arr+'_per_class'];assert len(vals)==20 and all(math.isfinite(x) and 0<=x<=1 for x in vals)
    for suffix,expected in [('target',vals[14]),('non_target',sum(x for i,x in enumerate(vals) if i!=14)/19),('all',sum(vals)/20)]:
     if m.get(key+'_'+suffix) is not None:assert math.isclose(m[key+'_'+suffix],expected,abs_tol=1e-12)
   a=item['checks'][arm];b=phases[d]['arms'][arm]
   assert a['last_sha256']==b['last_sha256'] and a['best_sha256']==b['best_sha256'] and b['last_primary_embedded_csv_rows']==200
   if not a['historical_clean_reuse']:
    assert b['dataset_output_hashes_exact']==16551 and b['optimizer_scheduler_scaler_rng_complete']
  a=p['metrics']['clean'];b=p['metrics']['S4'];P=None if a['mAP50_target']==0 else 1-b['mAP50_target']/a['mAP50_target'];U=None if a['mAP50_non_target']==0 else b['mAP50_non_target']/a['mAP50_non_target'];S=None if P is None or U is None or P+U==0 else 2*P*U/(P+U)
  damages=sorted([(x-y)/x for i,(x,y) in enumerate(zip(a['AP50_per_class'],b['AP50_per_class'])) if i!=14 and x!=0],reverse=True)
  scores=dict(P=P,U=U,S=S,WCD=damages[0] if len(damages)==19 else None,TCD3=sum(damages[:3])/3 if len(damages)==19 else None)
  for k,val in scores.items():assert (val is None and p['scores'][k] is None) or math.isclose(val,p['scores'][k],abs_tol=1e-12)
  summaries.append(dict(defense=d,clean_AP=a['mAP50_target'],S4_AP=b['mAP50_target'],clean_NT=a['mAP50_non_target'],S4_NT=b['mAP50_non_target'],**scores))
 prior=json.loads(load(W/'execution/OOM_RECOVERY_REPORT_COMPONENT_VERIFIED.json')['stdout'])
 assert evidence['source_components']['scripts/s4_pipeline_reports.py']['sha256']==prior['reporter_sha256']
 assert evidence['source_components']['scripts/s4_pipeline_frozen_report.py']['sha256']==prior['report_adapter_sha256']
 assert any('content' in x for x in evidence['source_components'].values()),'Missing separate reporting source receipt'
 assert len(list(csv.DictReader((DOC/'PHASE_A_PER_CLASS.csv').open())))==160
 assert len(list(csv.DictReader((DOC/'PHASE_B_PER_CLASS.csv').open())))==160
 DEST.mkdir(exist_ok=True)
 def sanitize(text):
  return re.sub(r'/project/[^\s\"\'<>),;]+', '<REMOTE_PATH>',text)
 for f in DOC.iterdir():
  if f.is_file(): (DEST/f.name).write_text(sanitize(f.read_text(encoding='utf-8')),encoding='utf-8')
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 accepted=dict(state='LOCAL_ACCEPTANCE_PASSED',UTC=stamp,pipeline_run_id=cfg['pipeline_run_id'],science_commit=cfg['science_commit'],launcher_commit='58f8c1cb3032abf4f296dbbce84ccb9d7a61f920',reporting_commit='05c73cf317ff16acd0e6e5eb960a858ba7b8c8d0',config_sha256=cfg['config_sha256'],eight_pairs=summaries,phase_A_per_class_rows=160,phase_B_per_class_rows=160,original_server_archive_sha256=sha(DOC/'ARCHIVE_MANIFEST.json'),all_original_archive_files_verified=True,remote_checkpoint_CSV_reverification=True,dataset_hash_audit_evidence='server accepted full16551 file audits; manifest hashes independently reread',GPU_forwards=0,AT='NOT_RUN',source_components=evidence['source_components'])
 (W/'execution/FINAL_LOCAL_ACCEPTANCE_20261022.json').write_text(json.dumps(accepted,indent=2),encoding='utf-8')
 (DEST/'LOCAL_ACCEPTANCE.json').write_text(sanitize(json.dumps(accepted,indent=2)),encoding='utf-8')
 manifest={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='PUBLICATION_MANIFEST.json'}
 (DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(dict(files=manifest,note='Published copies sanitize server paths. Original server archive hashes verified before sanitization; publication hashes apply to these copies.'),indent=2),encoding='utf-8')
 print(json.dumps({'accepted':True,'directory':str(DEST),'pairs':summaries,'publication_files':len(manifest)+1},ensure_ascii=False))

if __name__=='__main__':main()
