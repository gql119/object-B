"""One server-owned, fail-closed serial workflow with immutable phase releases.

Later data-dependent phases are registered only after their own implementation
and scientific gates close. A missing registration is a visible gate, never an
implied pass. The controller itself is never hot-patched and never retries a job.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .runtime import DATA_ROOT, ROOT, atomic, commit, load, now, require_review, sha, verify_release, write_once

ORDER = ('teacher200', 'teacher_eval', 'calibration', 'fourier24', 'export25k',
         'clean200', 's4_200', 'official_eval', 'final_report')


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    p = Path('/proc')/str(pid)
    return dict(pid=pid, start_ticks=int((p/'stat').read_text().rsplit(')', 1)[1].split()[19]), uid=p.stat().st_uid)


def live(ident):
    try:
        p = Path('/proc')/str(ident['pid'])
        parts = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return p.stat().st_uid == ident['uid'] and int(parts[19]) == ident['start_ticks'] and parts[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def capacity(root):
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/2**20
    disk = shutil.disk_usage(root).free/2**30
    def ticks():
        return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    io = []
    for _ in range(3):
        a = ticks(); time.sleep(1); b = ticks(); delta = [y-x for x, y in zip(a, b)]
        io.append(100*delta[4]/max(1, sum(delta)))
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True, timeout=20)
    free = {x.split(',')[0].strip(): int(x.split(',')[1]) for x in p.stdout.strip().splitlines()}['0']
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    r = dict(UTC=now(), RAM_available_GiB=mem, disk_free_GiB=disk, GPU_free_MiB=free,
             IOwait_percent=io, GPU_compute_pids=apps, CPU_load_is_advisory=True)
    r['admitted'] = mem >= 32 and disk >= 150 and free >= 24000 and not all(v >= 20 for v in io) and not apps
    return r


def predecessor(config):
    p = config['predecessor']
    s = load(Path(p['output'])/'pipeline_state.json')
    if s['pipeline_run_id'] != p['run_id'] or s['source_commit'] != p['source_commit']:
        raise ValueError('Wrong predecessor identity')
    if s['state'] != 'BLOCKED_PROTOCOL_APPROVAL' or s['active_job'] is not None:
        raise ValueError('Previous formal pipeline not complete')
    for stage in ('cutout', 'cutmix', 'mixup'):
        stage_state = s['stages'][stage]
        if isinstance(stage_state, dict):
            stage_state = stage_state.get('state')
        if stage_state != 'ACCEPTED':
            raise ValueError('Previous pair not accepted: ' + stage)
    if live(s['controller']):
        raise ValueError('Previous controller is live; inspect only')
    if sha(config['predecessor_acceptance']) != config['predecessor_acceptance_sha256']:
        raise ValueError('Independent old-three acceptance receipt drift')
    # Scan this owner's command lines only; do not touch any other user's tasks.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if any(t in cmd for t in ('coco_s4.phase_', 'robustness_s4_extended.worker', 'run_isolated_victim')):
                raise ValueError('Own existing scientific job conflicts with admission')
        except (FileNotFoundError, ProcessLookupError):
            pass
    return dict(state='ACCEPTED', native_state_sha256=sha(Path(p['output'])/'pipeline_state.json'), UTC=now())


def run_job(out, name, argv, cwd, status):
    job = out/'jobs'/name
    job.mkdir(parents=True, exist_ok=False)
    write_once(job/'intent.json', dict(argv=argv, cwd=str(cwd), UTC=now(), owner=identity()))
    status.update(state='RUNNING', active_job=name, updated_UTC=now())
    atomic(out/'pipeline_state.json', status)
    with (job/'console.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        ident = identity(child.pid)
        write_once(job/'child.json', ident)
        while child.poll() is None:
            status.update(active_child=ident, updated_UTC=now())
            atomic(out/'pipeline_state.json', status)
            time.sleep(10)
        result = dict(returncode=child.returncode, child=ident, ended_UTC=now())
        write_once(job/'exit.json', result)
    if child.returncode:
        raise RuntimeError('Scientific child failed; no automatic retry: ' + name)
    status.update(active_job=None, active_child=None)
    return result


def main():
    import fcntl
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    release = verify_release(a.release)
    cfg = load(a.config)
    if cfg['release_sha256'] != sha(a.release) or cfg['source_commit'] != commit():
        raise ValueError('Controller configuration/release identity mismatch')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)  # RRCTL creates its fresh bound output
    if (out/'pipeline_state.json').exists():
        raise FileExistsError('Existing controller binding is inspect-only; never restart it')
    lock = (out/'single_controller.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'receipts').mkdir(exist_ok=False)
    status = dict(schema='s4.coco.serial.v1', state='PREPARING', run_id=cfg['run_id'],
                  source_commit=commit(), config_sha256=sha(a.config), release_sha256=sha(a.release),
                  controller=identity(), stages={}, current_stage='teacher_smoke', active_job=None,
                  active_child=None, OP_enabled=False, phase_order=list(ORDER), created_UTC=now())
    atomic(out/'pipeline_state.json', status)
    try:
        write_once(out/'PREDECESSOR_ACCEPTANCE.json', predecessor(cfg))
        while True:
            r = capacity(out)
            atomic(out/'capacity_latest.json', r)
            if r['admitted']:
                break
            status.update(state='WAITING_RESOURCES', updated_UTC=now())
            atomic(out/'pipeline_state.json', status); time.sleep(300)
        smoke = out/'teacher_smoke'
        run_job(out, 'teacher_smoke', [sys.executable, '-B', '-m', 'coco_s4.phase_teacher', '--release', str(a.release),
                                     '--output', str(smoke), '--smoke-batches', '10'], ROOT, status)
        s = load(smoke/'smoke_summary.json')
        if s['state'] != 'passed' or not s['finite_loss'] or s['total_actual_batches'] != 10:
            raise ValueError('Teacher current-candidate GPU smoke not accepted')
        status.update(state='PENDING_SCIENTIFIC_REVIEW', current_stage='teacher200', updated_UTC=now())
        atomic(out/'pipeline_state.json', status)
        review = out/'receipts/TEACHER_SCIENTIFIC_CLOSURE.json'
        while not review.exists():
            time.sleep(60)
        require_review(review, a.release, smoke/'smoke_summary.json')
        while True:
            r = capacity(out); atomic(out/'capacity_latest.json', r)
            if r['admitted']:
                break
            status.update(state='WAITING_RESOURCES', updated_UTC=now())
            atomic(out/'pipeline_state.json', status); time.sleep(300)
        teacher = out/'teacher200'
        run_job(out, 'teacher200', [sys.executable, '-B', '-m', 'coco_s4.phase_teacher', '--release', str(a.release),
                                  '--output', str(teacher), '--review', str(review),
                                  '--smoke-evidence', str(smoke/'smoke_summary.json')], ROOT, status)
        receipt = load(teacher/'accepted.json')
        if receipt['state'] != 'ACCEPTED' or receipt['completed_epochs'] != 200:
            raise ValueError('Teacher formal acceptance failed')
        status['stages']['teacher200'] = dict(state='ACCEPTED', receipt_sha256=sha(teacher/'accepted.json'))
        # Each future phase release is immutable before execution. This supports
        # topological implementation/validation without changing a live Teacher.
        for phase in ORDER[1:]:
            status.update(state='WAITING_PHASE_RELEASE', current_stage=phase, updated_UTC=now())
            atomic(out/'pipeline_state.json', status)
            registration = out/'receipts'/(phase+'_READY.json')
            while not registration.exists():
                time.sleep(60)
            req = load(registration)
            if req.get('phase') != phase or req.get('workflow_run_id') != cfg['run_id'] or req.get('pre_run_result') != 'pass':
                raise ValueError('Invalid next-phase registration')
            source = Path(req['source_root']).resolve()
            if not source.is_relative_to(Path(cfg['owned_release_root']).resolve()):
                raise ValueError('Phase release escaped owned workflow root')
            if req['previous_stage_receipt_sha256'] != status['stages'][ORDER[ORDER.index(phase)-1]]['receipt_sha256']:
                raise ValueError('Next phase not bound to accepted previous stage')
            if sha(source/req['release_relative']) != req['release_sha256']:
                raise ValueError('Next phase immutable release differs')
            argv = [sys.executable, '-B', '-m', req['module'], '--release', str(source/req['release_relative']),
                    '--output', str(out/phase), '--registration', str(registration)]
            if not req['module'].startswith('coco_s4.phase_'):
                raise ValueError('Only explicit scientific phase entrypoints allowed')
            while True:
                r = capacity(out); atomic(out/'capacity_latest.json', r)
                if r['admitted']:
                    break
                time.sleep(300)
            run_job(out, phase, argv, source, status)
            accepted = out/phase/'accepted.json'
            if load(accepted)['state'] != 'ACCEPTED':
                raise ValueError('Next phase automatic acceptance failed')
            status['stages'][phase] = dict(state='ACCEPTED', receipt_sha256=sha(accepted))
        status.update(state='COMPLETED_AND_ACCEPTED', current_stage=None, updated_UTC=now())
    except BaseException as exc:
        status.update(state='FAILED', error=repr(exc), ended_UTC=now())
        raise
    finally:
        atomic(out/'pipeline_state.json', status)


if __name__ == '__main__':
    main()
