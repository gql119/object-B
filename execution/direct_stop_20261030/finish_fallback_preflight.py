"""Finite local client for three current-source GPU gates, no formal launch."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE.parent))
from at_cuda_rrctl_20261029 import services
ctl,_=services()


def cli(*args):
    r=subprocess.run([sys.executable,'-B',str(HERE/'current_source_probe.py'),*args],capture_output=True,text=True,encoding='utf8')
    print(json.dumps(dict(command=args,exit=r.returncode,stdout=r.stdout[-3000:],stderr=r.stderr[-3000:])),flush=True)
    if r.returncode:raise RuntimeError('Current-source gate command failed')


def finish(tag):
    end=time.monotonic()+900
    while True:
        state=ctl.inspect(tag)['status']['state']
        if state in ('workload_complete','completed'):
            cli('collect','--tag',tag);return
        if state in ('failed','aborted'):raise RuntimeError('Current-source gate terminal '+state+': '+tag)
        if time.monotonic()>end:raise TimeoutError('No further dispatch; bounded current gate still requires diagnosis')
        time.sleep(20)


if __name__=='__main__':
    finish('s4_at_selected_native_20261030_gpu01')
    for kind,tag in [('baseline','s4_at_selected_baseline_20261030_gpu01'),('resume','s4_at_selected_phase_20261030_gpu01')]:
        cli('launch','--tag',tag,'--kind',kind);finish(tag)
    print('CURRENT_SOURCE_FALLBACK_GPU_GATES_FINISHED_NO_FORMAL_LAUNCH',flush=True)
