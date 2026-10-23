"""Bounded real CPU-only supervision smoke, independent of the running Full job."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .common import ROOT,CONFIG,load,sha,atomic,commit

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=load(CONFIG)
    if not a.output.is_relative_to(Path(c['root'])/'outputs'):raise ValueError('New root only')
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    # Thirty CPU contracts use four synthetic images only, no model loss/forward or VOC reads.
    log=a.output/'CPU_TEST_CONSOLE.txt'
    with log.open('w') as f:
        subprocess.run([sys.executable,'-B','-m','robustness_s4_extended.validate_cpu'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    tests=load(ROOT/'robustness_s4_extended/reports/CPU_TEST_RESULT.json')
    # validate_cpu writes a runtime report in this isolated CPU-smoke checkout.
    # Reports are not computation-source hashes; the main waiter uses a separate clean checkout.
    if tests['status']!='PASSED' or tests['GPU_steps']!=0:raise ValueError('CPU contracts failed')
    from .successor import Successor
    from scripts.run_s4_robustness_pipeline import load as maybe,identity
    simulation=json.loads(json.dumps(c));simulation['root']=str(a.output);simulation['output_root']=str(a.output/'outputs'/'simulation');simulation['poll_seconds']=.1
    cfg=a.output/'simulation.json';atomic(cfg,simulation);s=Successor(cfg)
    import fcntl
    lockpath=s.root/'pipeline.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        probe=subprocess.run([sys.executable,'-B','-c',"import fcntl,sys;f=open(sys.argv[1],'a');\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:sys.exit(42)\nsys.exit(1)",str(lockpath)],cwd=ROOT)
        if probe.returncode!=42:raise ValueError('Singleton lock failed')
        counter=a.output/'child_count.txt'
        code="from pathlib import Path;import sys; p=Path(sys.argv[1]);p.write_text(str(int(p.read_text())+1) if p.exists() else '1')"
        argv=[sys.executable,'-B','-c',code,str(counter)]
        result=s.run_job('one_cpu_child',argv)
        t=Successor(cfg);adopted=t.run_job('one_cpu_child',argv)
        if result['returncode']!=0 or adopted['returncode']!=0 or counter.read_text()!='1':raise ValueError('Completed job duplicated')
        child=maybe(s.root/'jobs/one_cpu_child/child.json');scope=maybe(s.root/'jobs/one_cpu_child/scope_launch.json')
        if not child['memory_scope'] or child['memory_scope']['path']==scope['supervisor_scope']['path']:raise ValueError('Child not in independent ordinary scope')
    atomic(a.output/'smoke_summary.json',dict(state='passed',source_commit=commit(),config_sha256=sha(CONFIG),CPU_tests=tests['tests'],CPU_failures=tests['failures'],CPU_errors=tests['errors'],GPU_steps=0,real_VOC_reads=0,singleton_flock=True,production_supervisor=True,completed_job_executions=1,completed_job_adopted=True,ordinary_independent_child_scope=True,OOM_policy_overrides=False,existing_training_modified=False,child_identity=child,deleted_paths=[]))
    print(json.dumps(dict(state='passed',CPU_tests=tests['tests'],GPU_steps=0,source_commit=commit())))
if __name__=='__main__':main()
