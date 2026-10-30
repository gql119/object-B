"""Fresh serial AT generation; old global lock/terminal/evidence are retained.

The per-host flock is held for this process lifetime. A bounded CPU-only mode
exercises the same serial executor and lock without models/GPU or formal gates.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from robustness_s4_at.common import atomic,commit,load,now,protocol,sha,write_once
OLD='s4_at_serial_seed0_20261029_v2'

def live(pid,start):
    try:
        p=Path('/proc')/str(pid);r=(p/'stat').read_text().rsplit(')',1)[1].split()
        return p.stat().st_uid==1014 and int(r[19])==start and r[0]!='Z'
    except FileNotFoundError:return False

def old_terminal(owned):
    control=owned/'control'/OLD
    b=load(control/'binding.json');s=load(control/'status.json')
    if s['state']!='aborted' or b['commit']!='51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb':
        raise ValueError('Explicitly abandoned original must be genuinely aborted')
    if live(b['executor_pid'],b['executor_start_ticks']) or live(b['workload_pid'],b['workload_start_ticks']):
        raise ValueError('Original controller remains alive')
    if live(3339007,389475899):raise ValueError('Original scientific child remains alive')
    return dict(run_id=OLD,terminal=s['state'],binding_sha256=sha(control/'binding.json'))

def acquire(path):
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=1014:raise ValueError('Unowned serial guard')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return fd
    except BaseException:os.close(fd);raise

def serial(out,order,argv_for,receipt_for,cpu=False):
    state=dict(state='RUNNING',source_commit=commit(),stages={},current_run=None,UTC=now(),CPU_control_smoke=cpu)
    atomic(out/'pipeline_state.json',state)
    try:
        initial=None
        for arm in order:
            if not cpu:
                from robustness_s4_at.queue import resource_status
                resources=resource_status()
                while not resources['admitted']:
                    state.update(state='PENDING_RESOURCES',resources=resources,UTC=now())
                    atomic(out/'pipeline_state.json',state);time.sleep(300);resources=resource_status()
            state.update(state='RUNNING',current_run=arm,UTC=now());atomic(out/'pipeline_state.json',state)
            with (out/(arm+'.console.log')).open('xb') as log:
                code=subprocess.run(argv_for(arm),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            receipt=receipt_for(arm)
            if code or not receipt.is_file():raise RuntimeError('Arm failed; no automatic retry: '+arm)
            value=load(receipt)
            if value['state']!=('CPU_ACCEPTED_ONLY' if cpu else 'ACCEPTED'):raise ValueError('Invalid arm receipt')
            if not cpu:
                if value['run_id']!=arm or value['source_commit']!=commit() or value['epochs']!=200:raise ValueError('Arm source/epoch identity mismatch')
                current=value['initial_model_sha256']
                if initial is None:initial=current
                elif initial!=current:raise ValueError('Four arms did not share fresh initialization')
            state['stages'][arm]=dict(state=value['state'],receipt_sha256=sha(receipt))
        if not cpu:
            from robustness_s4_at.report import aggregate
            aggregate(out)
        state.update(state='CPU_CONTROL_PASSED' if cpu else 'COMPLETED_AND_ACCEPTED',current_run=None,UTC=now())
    except BaseException as exc:
        state.update(state='FAILED',error=repr(exc),UTC=now());raise
    finally:
        atomic(out/'pipeline_state.json',state)
        atomic(out/'CONTROLLER_TERMINAL.json',state)
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--preflight',type=Path);parser.add_argument('--cpu-control-smoke',action='store_true')
    args=parser.parse_args();p=protocol();owned=Path(p['remote_owned_root']).resolve()
    if os.getuid()!=1014 or commit()!=args.source_commit:raise ValueError('Wrong source/UID')
    relative=Path(__file__).resolve().relative_to(ROOT).as_posix()
    committed=subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)
    if hashlib.sha256(committed).hexdigest()!=sha(__file__):raise ValueError('Queue is not exact committed entry')
    allowed=owned/('validation' if args.cpu_control_smoke else 'outputs')/args.run_id/'result'
    if args.out.resolve()!=allowed:raise ValueError('New queue output is not an isolated exact RunID root')
    old=old_terminal(owned)
    if not args.cpu_control_smoke:
        if args.preflight is None:raise ValueError('Formal queue needs current-source scientific gate')
        from robustness_s4_at.worker import gate
        gate(args.preflight,args.source_commit)
    fd=acquire(str(owned/'AT_FRESH_SERIAL_ACTIVE.flock'))
    try:
        args.out.mkdir(parents=True,exist_ok=False)
        own=dict(run_id=args.run_id,pid=os.getpid(),uid=os.getuid(),source_commit=commit(),UTC=now(),
                 old_superseded=old,old_global_lock_retained=True)
        write_once(args.out/'CONTROLLER_LOCK.json',own)
        if args.cpu_control_smoke:
            old_lock=owned/'AT_SERIAL_CONTROLLER_LOCK.json';before=sha(old_lock)
            child="import json,sys,time;from pathlib import Path;p=Path(sys.argv[1]);p.mkdir(exist_ok=False);start=time.time();time.sleep(.1);(p/'accepted.json').write_text(json.dumps(dict(state='CPU_ACCEPTED_ONLY',start=start,end=time.time())))"
            order=['cpu_a','cpu_b','cpu_c']
            def argv(arm):return [sys.executable,'-B','-c',child,str(args.out/arm)]
            # A second process must fail to acquire the same held guard.
            contender="import fcntl,os,sys;f=os.open(sys.argv[1],os.O_RDWR|os.O_NOFOLLOW);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            r=subprocess.run([sys.executable,'-B','-c',contender,str(owned/'AT_FRESH_SERIAL_ACTIVE.flock')],capture_output=True)
            if not r.returncode:raise ValueError('Competing serial controller was admitted')
            state=serial(args.out,order,argv,lambda arm:args.out/arm/'accepted.json',True)
            rows=[load(args.out/arm/'accepted.json') for arm in order]
            if not all(a['end']<=b['start'] for a,b in zip(rows,rows[1:])):raise ValueError('CPU stages overlapped')
            if sha(old_lock)!=before:raise ValueError('Old global lock changed')
            write_once(args.out/'CPU_SERIAL_CONTROL_PROOF.json',dict(state='PASS_REAL_QUEUE_CONTROL_ONLY',
                source_commit=commit(),queue_entry_sha256=sha(__file__),duplicate_controller_rejected=True,
                three_stages_serial=True,old_global_lock_SHA_unchanged=True,models=0,GPU_work=False,
                terminal=state['state'],deleted_paths=[]))
        else:
            def argv(arm):return [sys.executable,'-B','-m','robustness_s4_at.worker','--formal','--run-id',arm,
                '--device','0','--out',str(args.out/arm),'--source-commit',args.source_commit,'--preflight',str(args.preflight.resolve())]
            serial(args.out,p['order'],argv,lambda arm:args.out/arm/'accepted.json')
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

if __name__=='__main__':main()
