"""Production-supervisor smoke using synthetic commands only; no CUDA or data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_s4_robustness_pipeline import Pipeline,ORDER,atomic,load,classify

def command(counter,delay=.05):
    return [sys.executable,'-B','-c',"import pathlib,sys,time;time.sleep(float(sys.argv[2]));p=pathlib.Path(sys.argv[1]);p.open('a').write('once\\n')",str(counter),str(delay)]

class Demo(Pipeline):
    crash=False
    def progress(self,jobdir):
        atomic(self.root/'pipeline_progress.json',dict(stage=self.state['current_stage'],time=time.time(),synthetic=True))
        if self.crash:os._exit(24)
    def action(self,key,action,**kwargs):return self.run_job(key,command(self.root/'counter.txt'))
    def existing(self):self.action('existing','synthetic')
    def stage(self,d):self.action(d,'synthetic')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--crash-controller',action='store_true');a=p.parse_args()
    root=a.output
    config=root/'demo_config.json'
    if a.crash_controller:
        demo=Demo(config);demo.crash=True;demo.run_job('orphan-demo',command(root/'orphan_counter.txt',2));return
    root.mkdir(parents=True,exist_ok=True)
    c=dict(pipeline_run_id=root.name,root=str(root.parent.parent),output_root=str(root/'queue'),order=ORDER,AT_authorized=False,config_sha256='synthetic_only',science_commit='synthetic_no_model',python=sys.executable,poll_seconds=.05,monitor_seconds=900)
    # The normal scheduler requires every output under the authorized outputs/.
    c['root']=str(root.parent.parent);c['output_root']=str(root/'queue')
    atomic(config,c)
    first=Demo(config);assert first.run()==0
    assert [k for k,v in first.state['stages'].items() if v['state']=='ACCEPTED']==ORDER
    before=(first.root/'counter.txt').read_text();assert len(before.splitlines())==len(ORDER)+1
    assert Demo(config).run()==0 and (first.root/'counter.txt').read_text()==before
    import fcntl
    with (first.root/'pipeline.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert Demo(config).run()==0
        assert (first.root/'counter.txt').read_text()==before
    crash=subprocess.run([sys.executable,'-B',__file__,'--output',str(root),'--crash-controller'])
    assert crash.returncode==24
    state=load(first.root/'pipeline_state.json');ident=state['jobs']['orphan-demo']['supervisor']
    time.sleep(.2)
    recovered=Demo(config);recovered.run_job('orphan-demo',command(root/'orphan_counter.txt',2))
    assert recovered.state['jobs']['orphan-demo']['supervisor']==ident
    assert (root/'orphan_counter.txt').read_text().splitlines()==['once']
    assert classify('CUDA out of memory')=='RETRYABLE_FAILURE'
    assert classify('AssertionError: non-finite raw loss; CUDA out of memory')=='BLOCKED'
    summary=dict(state='passed',GPU_forwards=0,training_batches=0,official_metrics_disabled=True,artifact_ingest_disabled=True,queue_order_exact=ORDER,completed_stages_never_rerun=True,flock_duplicate_prevented=True,controller_crash_exit=24,existing_supervisor_adopted=True,orphan_command_executions=1,child_survives_controller_exit=True,deleted_paths=[])
    atomic(root/'smoke_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
