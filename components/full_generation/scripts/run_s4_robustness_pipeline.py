"""Server-resident serial scheduler. Scientific computation stays at frozen e17c.

RRCTL owns the detached tmux session. This scheduler needs no client connection.
An OS flock guards the entire queue; independent supervisors survive scheduler
failure and expose durable PID/start-tick and exit receipts for safe adoption.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ORDER=['existing','gray','bdr','phase_A','gauss','meanf','medianf','data_aug','phase_B','archive']

def load(p,default=None):return json.loads(Path(p).read_text()) if Path(p).is_file() else default
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    with tmp.open('w') as f:
        json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def identity(pid):
    try:
        p=Path('/proc')/str(pid);raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        if parts[0]=='Z':return None
        return dict(pid=int(pid),start_ticks=int(parts[19]),uid=p.stat().st_uid)
    except (OSError,ValueError):return None
def same(record):
    now=identity(record.get('pid',-1)) if record else None
    return bool(now and now['uid']==os.getuid() and now['start_ticks']==record.get('start_ticks'))
def memory_scope(record):
    """Cheap observation only; never changes a limit or OOM policy."""
    if not same(record):return None
    try:
        rel=Path('/proc',str(record['pid']),'cgroup').read_text().strip().split('0::',1)[1]
        root=Path('/sys/fs/cgroup')/rel.lstrip('/')
        return dict(path=rel,**{name:(root/name).read_text().strip() for name in
            ['memory.current','memory.peak','memory.events','memory.pressure'] if (root/name).is_file()})
    except (OSError,IndexError):return None

def scope_argv(jobdir,argv,cwd):
    unit='s4rb-job-'+hashlib.sha256(str(Path(jobdir).resolve()).encode()).hexdigest()[:24]
    # A separate ordinary scope limits the blast radius, not the OOM policy.
    # No oom_score_adj, ManagedOOM*, memory limit or priority override is used.
    return unit, ['systemd-run','--user','--scope','--quiet','--unit='+unit,
        sys.executable,'-B',str(Path(__file__).resolve()),'--scope-child',str(jobdir),
        '--cwd',str(cwd),'--',*argv]

def scope_child(jobdir,argv,cwd):
    ident=identity(os.getpid());assert ident and ident['uid']==os.getuid()
    atomic(Path(jobdir)/'child.json',dict(**ident,argv=argv,cwd=str(cwd),started_at=time.time(),memory_scope=memory_scope(ident)))
    os.chdir(cwd);os.execvpe(argv[0],argv,os.environ.copy())
def classify(text):
    lower=text.lower()
    if any(x in lower for x in ['assertionerror','non-finite raw','nonfinite raw','persistent','label mismatch','checkpoint corrupt','scientific acceptance','invalid image']):return 'BLOCKED'
    if any(x in lower for x in ['out of memory','oom-kill','no space left','input/output error','resource temporarily unavailable','temporarily unavailable','stale file handle','connection reset']):return 'RETRYABLE_FAILURE'
    return 'FAILED'

class Blocked(RuntimeError):pass
class JobFailure(RuntimeError):
    def __init__(self,key,text):super().__init__(text);self.key=key;self.kind=classify(text)

def supervise(jobdir,argv,cwd,isolated=False):
    """Only this helper owns child Popen; never retries or alters arguments."""
    jobdir=Path(jobdir)
    with (jobdir/'stdout.log').open('ab',buffering=0) as out,(jobdir/'stderr.log').open('ab',buffering=0) as err:
        unit=None
        command=argv
        if isolated:
            unit,command=scope_argv(jobdir,argv,cwd)
            atomic(jobdir/'scope_launch.json',dict(unit=unit+'.scope',argv=command,supervisor=identity(os.getpid()),supervisor_scope=memory_scope(identity(os.getpid())),OOM_policy_overrides=False))
        child=subprocess.Popen(command,cwd=cwd,stdout=out,stderr=err,env=os.environ.copy())
        child_id=identity(child.pid)
        if child_id is None:
            # Very short synthetic jobs may already have exited; wait still owns it.
            child_id=dict(pid=child.pid,start_ticks=None,uid=os.getuid())
        if not isolated:atomic(jobdir/'child.json',dict(**child_id,argv=argv,cwd=str(cwd),started_at=time.time()))
        rc=child.wait()
        if isolated:
            child_id=load(jobdir/'child.json',child_id)
            try:
                observed=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','Result','-p','ActiveState','-p','MemoryPeak','-p','ControlGroup'],capture_output=True,text=True,timeout=15)
                details=dict(returncode=observed.returncode,stdout=observed.stdout,stderr=observed.stderr,child_returncode=rc)
            except (OSError,subprocess.TimeoutExpired) as e:details=dict(observer_error=repr(e),child_returncode=rc)
            atomic(jobdir/'scope_result.json',details)
            if rc:err.write(('\nIsolated scope status: '+details.get('stdout','unavailable')+'\n').encode())
    atomic(jobdir/'exit.json',dict(returncode=rc,ended_at=time.time(),child=child_id))
    return rc

class Pipeline:
    def __init__(self,config_path):
        self.config_path=Path(config_path).resolve();self.c=load(config_path);self.root=Path(self.c['output_root'])
        assert self.c['order']==ORDER and self.c['AT_authorized'] is False
        assert self.root.is_relative_to(Path(self.c['root'])/'outputs')
        self.root.mkdir(parents=True,exist_ok=True)
        self.state=load(self.root/'pipeline_state.json',dict(schema='s4.robustness.pipeline-state.v1',pipeline_run_id=self.c['pipeline_run_id'],state='PENDING',current_stage='existing',stages={},jobs={},created_at=time.time(),config_sha256=self.c['config_sha256'],science_commit=self.c['science_commit']))
        assert self.state['pipeline_run_id']==self.c['pipeline_run_id']
        self.last_monitor=time.time();self.config_digest=sha(config_path);self.children={}
    def save(self):
        self.state['updated_at']=time.time();atomic(self.root/'pipeline_state.json',self.state)
    def event(self,event,**details):
        record=dict(time=time.time(),event=event,stage=self.state['current_stage'],**details)
        with (self.root/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        with (self.root/'pipeline.log').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    def worker(self,action,**kwargs):
        argv=[self.c['python'],'-B',str(HERE/'s4_pipeline_worker.py'),'--config',str(self.config_path),'--action',action]
        for key,value in kwargs.items():argv.extend(['--'+key.replace('_','-'),str(value)])
        return argv
    def run_job(self,key,argv,result=False):
        jobdir=self.root/'jobs'/key.replace(':','__');job=self.state['jobs'].get(key)
        if job is None:
            if jobdir.exists():raise Blocked('Unbound job directory collision: '+str(jobdir))
            jobdir.mkdir(parents=True)
            job=dict(state='LAUNCH_INTENT',directory=str(jobdir),argv=argv,created_at=time.time())
            self.state['jobs'][key]=job;self.save()
            isolation=['--isolated-scope'] if self.c.get('isolate_job_scopes',False) else []
            process=subprocess.Popen([self.c['python'],'-B',str(Path(__file__).resolve()),'--supervise',str(jobdir),'--cwd',str(HERE.parent),*isolation,'--',*argv],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.children[key]=process
            ident=identity(process.pid)
            if ident is None:ident=dict(pid=process.pid,start_ticks=None,uid=os.getuid())
            job.update(supervisor=ident,state='RUNNING');self.save();self.event('job_started',job=key,supervisor=ident)
        elif job['argv']!=argv:raise Blocked('Recovery argv differs for '+key)
        self.state['active_job']=key;self.save()
        time.sleep(.2)
        while not (jobdir/'exit.json').exists():
            if not same(job.get('supervisor')):
                child=load(jobdir/'child.json')
                if same(child):
                    self.state['observation']='supervisor lost; original child still running; adoption only';self.save()
                else:raise Blocked('Lost supervisor/child without exit receipt; no duplicate restart: '+key)
            self.progress(jobdir)
            time.sleep(self.c['poll_seconds'])
        receipt=load(jobdir/'exit.json');job.update(state='COMPLETED' if receipt['returncode']==0 else 'FAILED',exit=receipt);self.save()
        if key in self.children:self.children.pop(key).wait(timeout=10)
        if receipt['returncode']:
            text='\n'.join(p.read_text(errors='replace')[-24000:] for p in [jobdir/'stdout.log',jobdir/'stderr.log'] if p.exists())
            raise JobFailure(key,text)
        return load(jobdir/'result.json') if result else receipt
    def action(self,key,action,**kwargs):
        jobdir=self.root/'jobs'/key.replace(':','__')
        return self.run_job(key,self.worker(action,result=jobdir/'result.json',**kwargs),True)
    def resource(self):
        dest=self.root/'resource_latest.json'
        r=subprocess.run(self.worker('resource',result=dest),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        if r.returncode:raise Blocked('Resource observation unavailable; no unsafe admission')
        return load(dest)
    def admission(self):
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('frozen_resource_policy',Path(self.c['science_repo'])/'ue_framework/s4_robustness_resources.py');policy=module_from_spec(spec);spec.loader.exec_module(policy)
        while True:
            samples=[]
            for i in range(3):
                samples.append(self.resource())
                if i<2:time.sleep(30)
            blockers=[b for s in samples for b in policy.capacity_blockers(s)]
            if policy.sustained_iowait(samples):blockers.append('sustained I/O wait >=20%')
            atomic(self.root/'resource_admission.json',dict(samples=samples,blockers=blockers,passed=not blockers,policy_id=policy.POLICY_ID))
            if not blockers:self.state.update(state='RUNNING',resource_blockers=[]);self.save();return
            self.state.update(state='PAUSED_RESOURCE',resource_blockers=blockers);self.save();self.event('resource_wait',blockers=blockers);time.sleep(900)
    def progress(self,jobdir):
        progress=dict(pipeline_run_id=self.c['pipeline_run_id'],stage=self.state['current_stage'],time=time.time(),state=self.state['state'])
        child=load(jobdir/'child.json');progress['child']=child
        progress['child_memory_scope']=memory_scope(child);progress['controller_memory_scope']=memory_scope(identity(os.getpid()))
        if self.state.get('paused_child'):
            paused=self.state['paused_child']
            if not same(paused):raise Blocked('Recorded paused child identity changed')
            argv=paused['argv'];arm=Path(argv[argv.index('--output')+1])
            self.resume_io(paused,arm,self.state['paused_checkpoint_sha256'])
        if child:
            argv=child['argv']
            if '--output' in argv:
                arm=Path(argv[argv.index('--output')+1]);progress['training']=load(arm/'training_progress.json');progress['AMP']=load(arm/'amp_audit/latest.json')
                weights=arm/'victim_run/victim/weights';progress['checkpoint_files']={n:dict(size=p.stat().st_size,mtime=p.stat().st_mtime) for n in ['last.pt','best.pt'] if (p:=weights/n).is_file()}
        atomic(self.root/'pipeline_progress.json',progress)
        if time.time()-self.last_monitor>=self.c['monitor_seconds']:
            sample=self.resource();self.last_monitor=time.time();self.event('resource_telemetry',sample=sample)
            # CPU/GPU busy/presence remain speed-only telemetry. Only preserved
            # sustained I/O safety invokes STOP, only for this exact own child.
            if sample['io_wait_percent_sample']>=20 and child and '--phase' in child['argv'] and 'arm' in child['argv']:
                samples=[sample]
                for _ in range(2):time.sleep(30);samples.append(self.resource())
                if all(s['io_wait_percent_sample']>=20 for s in samples):self.pause_io(child,arm)
    def pause_io(self,child,arm):
        weights=arm/'victim_run/victim/weights/last.pt';p=load(arm/'training_progress.json')
        if not p or not weights.is_file() or not same(child):
            self.event('io_safety_action_unavailable',reason='no stable checkpoint/identity; no unsafe signal');return
        digest=sha(weights);time.sleep(1)
        if digest!=p['last_sha256'] or digest!=sha(weights) or not same(child):return
        self.state.update(state='PAUSED_RESOURCE',paused_child=child,paused_checkpoint_sha256=digest);self.save()
        os.kill(child['pid'],signal.SIGSTOP);self.event('own_child_paused_for_sustained_io',child=child,checkpoint_sha256=digest)
        self.resume_io(child,arm,digest)
    def resume_io(self,child,arm,digest):
        weights=arm/'victim_run/victim/weights/last.pt'
        while True:
            samples=[]
            for i in range(3):
                samples.append(self.resource())
                if i<2:time.sleep(30)
            clear=all(s['io_wait_percent_sample']<20 and s['mem']['MemAvailable_kib']*1024>=32*1024**3 and s.get('disk_free_bytes',s.get('disk',{}).get('free_bytes',0))>=150*1024**3 for s in samples)
            if clear:
                if not same(child) or sha(weights)!=digest:raise Blocked('Paused identity/checkpoint binding changed')
                os.kill(child['pid'],signal.SIGCONT);self.state.update(state='RUNNING',paused_child=None);self.save();self.event('own_child_resumed',child=child);return
            time.sleep(900)
    def existing(self):
        for d,rid in self.c['bootstrap_runs'].items():
            out=Path(self.c['root'])/'outputs'/rid
            while load(out/'pipeline_status.json',{}).get('state')!='complete':
                status=load(out/'pipeline_status.json',{});child=load(out/'active_child.json')
                if status.get('state')=='failed':raise Blocked('Existing '+d+' failed; preserve original binding')
                if child and child.get('active') is not False and not same(child):raise Blocked('Existing run process no longer matches without completion: '+d)
                atomic(self.root/'pipeline_progress.json',dict(stage='existing',defense=d,child=child,S4=load(out/'S4/training_progress.json'),clean=load(out/'clean/training_progress.json'),time=time.time()))
                time.sleep(self.c['poll_seconds'])
            answer=self.action('existing:'+d+':audit','audit',output=out,defense=d)
            self.receipt(rid,answer)
    def receipt(self,rid,answer):
        p=self.root/'execution/acceptance'/f'{rid}.json'
        if p.exists():assert load(p)['paired_results_sha256']==answer['paired_results_sha256']
        else:atomic(p,answer)
    def stage(self,d):
        spec=self.c['stages'][d];out=Path(spec['output']);s=self.state['stages'][d]
        if 'output_owned' not in s:
            if out.exists():raise Blocked('Unowned formal output collision '+str(out))
            s['output_owned']=True;self.save()
        self.action(d+':init','init',output=out,defense=d)
        selected=load(out/'selected_arms.json',{})
        for arm in ['S4','clean']:
            if arm in selected:continue
            valid=False
            if arm=='clean':
                check=self.action(d+':control-check','historical-check',defense=d);valid=check['valid'];s['clean_control']=check;self.save()
            arms=s.setdefault('arms',{});record=arms.setdefault(arm,dict(attempt=0,retry_count=0));self.save()
            while True:
                attempt=record['attempt'];label=f'{d}:{arm}:attempt{attempt+1:02d}';armout=out/'arm_attempts'/f'{arm}_attempt{attempt+1:02d}'
                record.update(run_id=spec['run_id']+f'__{arm}'+('' if attempt==0 else f'_recovery{attempt:02d}'),output=str(armout),exact_resume=False);self.save()
                # A completed training arm only requires missing evaluation; never
                # train again after 200 auditable rows have been preserved.
                rows_path=armout/'victim_run/victim/results.csv'
                rows=list(csv.DictReader(rows_path.open())) if rows_path.is_file() else []
                if len(rows)==200 and load(armout/'pipeline_status.json',{}).get('state')!='complete':
                    previous=load(armout/'pipeline_status.json',{})
                    if previous.get('state')=='failed' and classify(previous.get('error',''))=='BLOCKED':raise Blocked('Scientific failure after200rows must not be hidden by evaluation recovery')
                    self.admission();self.action(label+':evaluation-recovery','finalize',output=armout,defense=d)
                try:
                    if load(armout/'pipeline_status.json',{}).get('state')!='complete':
                        self.admission()
                        if valid:self.action(label,'historical',output=armout,defense=d)
                        else:
                            argv=[self.c['python'],'-B',str(Path(self.c['science_repo'])/'scripts/run_s4_robustness.py'),'--phase','arm','--output',str(armout),'--defense',d,'--arm',arm]
                            self.run_job(label,argv)
                    break
                except JobFailure as e:
                    record.update(state=e.kind,failure_reason=str(e)[-8000:]);self.save();self.event('arm_failure',arm=arm,kind=e.kind,run_id=record['run_id'])
                    if e.kind!='RETRYABLE_FAILURE' or record['retry_count']>=self.c['max_environment_retries']:raise
                    record['retry_count']+=1
                    # Retain a finished 200-epoch training and retry only evaluation.
                    rows=list(csv.DictReader(rows_path.open())) if rows_path.is_file() else []
                    if len(rows)==200:
                        time.sleep(self.c['backoff_seconds'][record['retry_count']-1]);self.admission()
                        self.action(label+f':evaluation-retry{record["retry_count"]}','finalize',output=armout,defense=d);break
                    record['attempt']+=1;record['recovery_reason']='environment failure; fresh seed0/full200 in new recovery output; DataLoader exact-resume unproven';self.save()
                    time.sleep(self.c['backoff_seconds'][record['retry_count']-1])
            link=out/arm
            if link.is_symlink():assert link.resolve()==armout.resolve()
            else:
                if link.exists():raise Blocked('Canonical arm collision')
                link.symlink_to(armout,target_is_directory=True)
            selected[arm]=dict(output=str(armout),run_id=record['run_id'],historical=valid,retry_count=record['retry_count'],exact_resume=False)
            atomic(out/'selected_arms.json',selected);record.update(state='COMPLETED');self.save();self.event('arm_completed',arm=arm,run_id=record['run_id'])
        if not (out/'paired_results.json').exists():self.action(d+':assemble','assemble',output=out,defense=d)
        answer=self.action(d+':audit','audit',output=out,defense=d);self.receipt(spec['run_id'],answer)
        s['checkpoint_hashes']=answer['arms'];s['acceptance_state']='ACCEPTED';self.save()
    def run(self):
        import fcntl
        lock=(self.root/'pipeline.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(existing_instance=True,state=load(self.root/'pipeline_state.json'))));return 0
        if self.state.get('orchestrator_config_sha256') not in (None,self.config_digest):raise Blocked('Pipeline config changed during recovery')
        self.state.update(orchestrator_config_sha256=self.config_digest,controller=identity(os.getpid()),state='RUNNING');self.save();self.event('controller_attached',pid=os.getpid())
        try:
            self.action('verify','verify')
            for name in ORDER:
                s=self.state['stages'].setdefault(name,dict(state='PENDING',retry_count=0))
                if s['state']=='ACCEPTED':continue
                self.state.update(current_stage=name,state='RUNNING');s.update(state='RUNNING',started_at=s.get('started_at',time.time()));self.save();self.event('stage_started')
                if name=='existing':self.existing()
                elif name in ('phase_A','phase_B','archive'):
                    phase={'phase_A':'A','phase_B':'B','archive':'FINAL'}[name]
                    self.action(name+':report','reports',phase=phase)
                else:self.stage(name)
                s.update(state='ACCEPTED',completed_at=time.time());self.save();self.event('stage_accepted')
            self.state.update(state='COMPLETED_AND_ACCEPTED',completed_at=time.time(),active_job=None);self.save();self.event('pipeline_completed_and_accepted');return 0
        except (Blocked,JobFailure,AssertionError,ValueError) as e:
            self.state.update(state='BLOCKED' if not isinstance(e,JobFailure) else e.kind,failure_reason=repr(e),failed_at=time.time());self.save();self.event('pipeline_stopped',error=repr(e));raise
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--status',action='store_true');p.add_argument('--supervise');p.add_argument('--scope-child');p.add_argument('--isolated-scope',action='store_true');p.add_argument('--cwd');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if a.scope_child:return scope_child(a.scope_child,command,a.cwd)
    if a.supervise:return supervise(a.supervise,command,a.cwd,a.isolated_scope)
    if not a.config:p.error('--config required')
    if a.status:
        c=load(a.config);root=Path(c['output_root']);print(json.dumps(dict(state=load(root/'pipeline_state.json'),progress=load(root/'pipeline_progress.json')),indent=2));return 0
    return Pipeline(a.config).run()

if __name__=='__main__':raise SystemExit(main())
