"""Independent durable successor. Waiting requires no CUDA, data traversal or client."""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
from .common import ROOT,CONFIG,atomic,load,sha,commit,verify_source
from .predecessor import full_state,verify_predecessor
from .selection import SelectionBlocked,select_and_lock,decision_identity
from scripts.run_s4_robustness_pipeline import Pipeline,Blocked,identity,same

class Successor(Pipeline):
    def __init__(self,config_path=CONFIG):
        self.config_path=Path(config_path);self.c=load(config_path);self.root=Path(self.c['output_root'])
        if not self.root.is_relative_to(Path(self.c['root'])/'outputs'):raise ValueError('Output boundary')
        self.root.mkdir(parents=True,exist_ok=True)
        self.state=load(self.root/'pipeline_state.json') if (self.root/'pipeline_state.json').exists() else dict(schema='s4.extended.pipeline-state.v1',pipeline_run_id=self.c['pipeline_run_id'],source_commit=commit(),config_sha256=sha(config_path),state='WAITING_PREVIOUS_PIPELINE',current_stage='predecessor',stages={},jobs={},active_job=None)
        if (self.state['source_commit'],self.state['config_sha256'],self.state['pipeline_run_id'])!=(commit(),sha(config_path),self.c['pipeline_run_id']):raise ValueError('Existing queue binding differs')
        self.config_digest=sha(config_path);self.children={};self.last_monitor=time.time()
    def save(self):
        if sha(self.config_path)!=self.config_digest:raise ValueError('Queue config changed during execution')
        super().save()
        atomic(self.root/'PIPELINE_STATUS.json',self.state)
    def progress(self,jobdir):
        child_path=jobdir/'child.json';child=load(child_path) if child_path.exists() else None
        atomic(self.root/'pipeline_progress.json',dict(stage=self.state['current_stage'],child=child,state=self.state['state'],time=time.time()))
    def command(self,action,defense=None,arm=None,output=None):
        a=[self.c['python'],'-B','-m','robustness_s4_extended.worker','--action',action]
        for k,v in [('defense',defense),('arm',arm),('output',output)]:
            if v is not None:a+=['--'+k,str(v)]
        return a
    def own_training_conflict(self):
        # Cheap /proc reads only after predecessor accepted, never idle-GPU inference.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:
                if p.stat().st_uid!=os.getuid():continue
                argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
                if not argv or 'python' not in Path(argv[0]).name:continue
                entries=[Path(x).name for x in argv[1:] if x.endswith('.py')]
                if any(x.startswith(('run_s4','run_s5','train')) for x in entries) or 'robustness_s4_extended.worker' in argv:
                    return dict(pid=int(p.name),identity=identity(int(p.name)),entry=entries)
            except (OSError,ValueError):continue
        return None
    def verify_wait_gate(self):
        p=self.c['predecessor'];s=full_state(self.c)
        if s and (s.get('run_id'),s.get('source_commit'),s.get('config_sha256'))!=(p['run_id'],p['source_commit'],p['config_sha256']):raise Blocked('Predecessor identity differs')
        if s and s.get('state')=='failed':raise SelectionBlocked('BLOCKED_FULL_VALIDATION: predecessor failed; never treat as inferior')
        return bool(s and s.get('state')=='complete')
    def report(self):
        rows=[];full=[];extended=[];all_metrics={}
        for name in ['cutout','cutmix','mixup']:
            p=self.root/'experiments'/name/'paired_results.json'
            if self.state['stages'].get(name)=='ACCEPTED':
                r=load(p);scores=dict(r['scores'],target_poison=r['scores']['target_S4'],NT_poison=r['scores']['NT_S4']);rows.append(dict(defense=name,selected_method=r['selected_method'],state='ACCEPTED',**{k:scores[k] for k in ['target_clean','target_poison','NT_clean','NT_poison','P','U','S','WCD','TCD3']}))
                for arm in ['clean','poison']:
                    m=r['metrics'][arm]
                    extended.append(dict(defense=name,selected_method=r['selected_method'],arm=arm,checkpoint_role='last.pt@200',**{k:v for k,v in m.items() if isinstance(v,(float,int))}))
                    all_metrics[name+'_'+arm]={role:load(self.root/'experiments'/name/arm/(role+'_evaluation.json')) for role in ['last_primary','best_secondary']}
                    for i in range(20):full.append(dict(defense=name,selected_method=r['selected_method'],arm=arm,class_id=i,**{k:r['metrics'][arm][k+'_per_class'][i] for k in ['AP50','AP75','AP50_95','Recall','Precision']}))
            else:rows.append(dict(defense=name,selected_method=self.state.get('selected_method'),state=self.state['stages'].get(name,'NOT_RUN'),**{k:None for k in ['target_clean','target_poison','NT_clean','NT_poison','P','U','S','WCD','TCD3']}))
        rows.append(dict(defense='op',selected_method=self.state.get('selected_method'),state='BLOCKED_PROTOCOL_APPROVAL',**{k:None for k in ['target_clean','target_poison','NT_clean','NT_poison','P','U','S','WCD','TCD3']}))
        reports=self.root/'reports';reports.mkdir(exist_ok=True)
        for file,data in [('EXTENDED_RESULTS.csv',rows),('EXTENDED_ROBUSTNESS_RESULTS.csv',rows),('EXTENDED_PER_CLASS_RESULTS.csv',full),('EXTENDED_ROBUSTNESS_PER_CLASS.csv',full),('EXTENDED_METRICS.csv',extended)]:
            if data:
                with (reports/file).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        atomic(reports/'REPORT_STATE.json',dict(pipeline_run_id=self.c['pipeline_run_id'],stages=self.state['stages'],selected_method=self.state.get('selected_method'),OP='BLOCKED_PROTOCOL_APPROVAL',source_commit=commit(),config_sha256=sha(CONFIG),single_seed_descriptive=True))
        atomic(reports/'METRICS_ALL_CHECKPOINTS.json',all_metrics)
        (reports/'EXTENDED_REPORT.md').write_text('# Extended S4 robustness\n\nTraining-time augmentation defenses: Cutout, CutMix, MixUp. Paired Clean/selected Poison; selected method and locked decision are recorded in every result. Clean validation, last@200 primary; best secondary. See results and full per-class tables. Unfinished cells are empty, never zero.\n\nOrthogonal Projection data-recovery defense: BLOCKED_PROTOCOL_APPROVAL. CPU mathematics only; no projection model or victim has been trained.\n\nSingle seed0 describes this protocol; AP alone does not demonstrate a feature or causal mechanism.\n')
        (reports/'EXTENDED_ROBUSTNESS_REPORT.md').write_text((reports/'EXTENDED_REPORT.md').read_text()+ '\nSelected Method: '+str(self.state.get('selected_method'))+'\n')
        atomic(reports/'ARCHIVE_MANIFEST.json',{p.name:sha(p) for p in reports.iterdir() if p.is_file() and p.name!='ARCHIVE_MANIFEST.json'})
    def run(self):
        import fcntl
        with (self.root/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            verify_source(self.c);self.state['controller']=identity(os.getpid());self.save()
            try:
                if not (self.root/'PREDECESSOR_ACCEPTANCE.json').exists():
                    while not self.verify_wait_gate():
                        self.state.update(state='WAITING_PREVIOUS_PIPELINE',current_stage='predecessor',active_job=None);self.save();time.sleep(self.c['poll_seconds'])
                    self.state.update(state='CHECK_PREVIOUS_ACCEPTANCE');self.save()
                    proof=verify_predecessor(self.c)
                    if not proof or proof['state']!='COMPLETED_AND_ACCEPTED':raise Blocked('Predecessor scientific acceptance unavailable')
                    atomic(self.root/'PREDECESSOR_ACCEPTANCE.json',proof)
                self.state.update(state='COMPARE_S4_VS_FULL',current_stage='selection');self.save()
                decision=select_and_lock(self.c)
                self.state.update(selected_method=decision['selected_method'],decision_sha256=sha(self.root/'S4_FULL_DECISION.json'));self.save()
                conflict=self.own_training_conflict()
                while conflict:
                    self.state.update(state='WAITING_OTHER_BOUND_WORK',conflict=conflict);self.save();time.sleep(self.c['poll_seconds']);conflict=self.own_training_conflict()
                self.state.update(state='VALIDATE_NEW_CODE',current_stage='validation');self.save()
                self.run_job('validate',self.command('validate'))
                for name in ['cutout','cutmix','mixup']:
                    self.state.update(state='PENDING_GPU_SMOKE',current_stage='smoke_'+name);self.save()
                    self.run_job('smoke_'+name,self.command('smoke',name,output=self.root/'experiments'/('smoke_'+name)))
                self.state.update(state='PENDING_SCIENTIFIC_REVIEW',current_stage='scientific_review',active_job=None);self.save()
                self.event('all_current_source_GPU_smokes_ready_for_single_scientific_review',**decision_identity(self.c))
                # This receipt is issued by the one scientific reviewer/main-agent closure after GPU evidence.
                # Never synthesize an allow_run result or call a reviewer without actual smoke.
                while not Path(self.c['review_receipt']).exists():time.sleep(self.c['poll_seconds'])
                from .worker import review_gate
                review_gate(self.c)
                for name in ['cutout','cutmix','mixup']:
                    if self.state['stages'].get(name)=='ACCEPTED':continue
                    self.state.update(state='RUNNING',current_stage=name);self.save()
                    for arm in ['clean','poison']:
                        self.run_job(name+'_'+arm,self.command('arm',name,arm,self.root/'experiments'/name/arm))
                    self.run_job('accept_'+name,self.command('accept',name,output=self.root/'experiments'/name))
                    proof=load(self.root/'experiments'/name/'ACCEPTANCE.json')
                    if proof['state']!='ACCEPTED':raise Blocked('Pair scientific acceptance failed')
                    self.state['stages'][name]='ACCEPTED';self.state['active_job']=None;self.save();self.report()
                self.state.update(state='BLOCKED_PROTOCOL_APPROVAL',current_stage='op',active_job=None);self.state['stages']['op']='BLOCKED_PROTOCOL_APPROVAL';self.save();self.report()
            except BaseException as e:
                self.state.update(state='BLOCKED_FULL_VALIDATION' if isinstance(e,SelectionBlocked) else 'FAILED',error=repr(e));self.save();self.report();raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--expected-source');a=p.parse_args();c=load(CONFIG)
    if a.status:
        path=Path(c['output_root'])/'pipeline_state.json';print(json.dumps(load(path) if path.exists() else {'state':'NOT_DEPLOYED'}));return
    if not a.expected_source or a.expected_source!=commit():raise ValueError('Exact committed source required')
    Successor().run()
if __name__=='__main__':main()
