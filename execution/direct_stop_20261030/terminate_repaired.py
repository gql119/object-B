"""Use a reviewed scoped RRCTL abort repair, with fresh pre/post evidence."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services,PYTHON
from guarded_abort_repair import CONTROL,OLD_TAG,AUTH_SHA,request,invoke
from stop_old import READ,VERIFY,AUTH,SOURCE,SPEC

def save(name,value):
    path=HERE/name
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inspect','terminate']);args=ap.parse_args()
    assert hashlib.sha256(AUTH.read_bytes()).hexdigest()==AUTH_SHA
    digest=hashlib.sha256(CONTROL.encode()).hexdigest()
    proof=json.loads((HERE/'s4_at_abort_exec_guard_20261030_cpu02/CPU_ABORT_MAIN_CLOSURE.json').read_text())
    assert proof['state']=='PASS_REAL_RRCTL_CPU_ABORT_WITH_POSTEXEC_GUARD' and proof['control_source_sha256']==digest
    if args.action=='terminate':
        review=json.loads((HERE/'TARGETED_CONTROL_REVIEW_20261030.json').read_text())
        assert review['allow_stop'] is True and review['control_source_sha256']==digest
    ctl,t=services();inspected=ctl.inspect(OLD_TAG);b=inspected['binding']
    assert inspected['status']['state']=='running' and b['commit']==SOURCE and b['run_spec_sha256']==SPEC
    raw=t.download(b['control_root']+'/binding.json')
    r=t.run([PYTHON,'-B','-c',READ,json.dumps(b)])
    assert not r.returncode,repr((r.stdout,r.stderr))
    before=json.loads(r.stdout);record=dict(binding=b,before=before,authorization_sha256=AUTH_SHA)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    pre=save(stamp+'_REPAIRED_ABORT_PRESTATE.json',record)
    req=request(b,raw);guard=invoke(t,PYTHON,req);save(stamp+'_PRODUCTION_READONLY_GUARD.json',guard)
    assert not guard['returncode'],guard
    if args.action=='inspect':
        print(json.dumps(dict(prestate=str(pre),guard=json.loads(guard['stdout']),progress={k:v for k,v in before['progress'].items() if k!='latest_PGD'})))
        return
    save(stamp+'_REPAIRED_TERMINATION_INTENT.json',dict(authorization=AUTH_SHA,guard_source_sha256=digest,
        review='TARGETED_CONTROL_REVIEW_20261030.json',prestate=str(pre),run_id=OLD_TAG,action='RRCTL existing abort with exact canonical postexec fingerprint'))
    req['action']='abort';result=invoke(t,PYTHON,req);save(stamp+'_REPAIRED_ABORT_RESULT.json',result)
    assert not result['returncode'],result
    terminal=ctl.inspect(OLD_TAG);assert terminal['status']['state']=='aborted'
    check=t.run([PYTHON,'-B','-c',VERIFY,json.dumps(record)])
    save(stamp+'_POST_ABORT_VERIFICATION_RAW.json',dict(returncode=check.returncode,
        stdout=check.stdout.decode() if isinstance(check.stdout,bytes) else check.stdout,
        stderr=check.stderr.decode() if isinstance(check.stderr,bytes) else check.stderr))
    assert not check.returncode
    final=dict(state='OLD_AT_TERMINATED',UTC=datetime.now(timezone.utc).isoformat(),run_id=OLD_TAG,
        old_completed=False,old_accepted=False,old_progress_abandoned_by_human=True,
        prestate=str(pre),RRCTL=terminal['status'],verification=json.loads(check.stdout),
        control_source_sha256=digest,binding_unmodified=True,checkpoint_retained=before['checkpoint'])
    path=save(stamp+'_OLD_AT_TERMINATED.json',final);print(json.dumps(dict(evidence=str(path),**final)))

if __name__=='__main__':main()
