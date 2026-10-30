"""Equivalent publication read via authenticated API after local Git TLS EOF.

Local orchestration only. Remote runtime remains the already published frozen
commit. Scientific source and every OriginalS4 hash are checked unchanged.
"""
import json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(ROOT))
import current_source_probe as entry
from robustness_s4_at.common import scientific_source,protocol,sha
from at_cuda_rrctl_20261029 import services
ctl,_=services()


def verified_current():
    proof=json.loads((HERE/'REMOTE_BRANCH_API_VERIFIED.json').read_text())
    observed=datetime.strptime(proof['observed_UTC'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)
    if not 0<=(datetime.now(timezone.utc)-observed).total_seconds()<=900:raise ValueError('Refresh exact authenticated branch-ref evidence before dispatch')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if proof['state']!='VERIFIED_AUTHENTICATED_GITHUB_API_BRANCH_REF' or proof['raw_ref']['ref']!='refs/heads/codex/s4-at-robustness-20261027' or proof['raw_ref']['object']['sha']!=source:
        raise ValueError('Actual authenticated branch SHA does not match runtime source')
    if scientific_source()['uncommitted_files']:raise ValueError('Scientific source changed')
    for rel,digest in protocol()['original_sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('OriginalS4 source drift')
    return source


def call(*args):
    prior=sys.argv;sys.argv=['current_source_probe.py',*args]
    try:entry.main()
    finally:sys.argv=prior


def finish(tag):
    end=time.monotonic()+900
    while True:
        state=ctl.inspect(tag)['status']['state']
        if state in ('completed','workload_complete'):call('collect','--tag',tag);return
        if state in ('failed','aborted'):raise RuntimeError('Finite current-source probe failed: '+tag)
        if time.monotonic()>end:raise TimeoutError('No further dispatch; inspect existing bounded run')
        time.sleep(20)


if __name__=='__main__':
    entry.current=verified_current
    for kind,tag in [('baseline','s4_at_selected_baseline_20261030_gpu01'),('resume','s4_at_selected_phase_20261030_gpu01')]:
        assert not (HERE/'current_probes'/tag).exists(),'Existing probe is inspect-only'
        call('launch','--kind',kind,'--tag',tag);finish(tag)
    print('VERIFIED_CURRENT_SOURCE_GPU_GATES_FINISHED',flush=True)
