"""Future single serial AT queue. No watcher, retries, signals, or launch at import."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import ROOT, protocol, load, write_once, atomic, sha, commit, now


def process_live(expected):
    try:
        p = Path('/proc') / str(expected['pid'])
        row = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return p.stat().st_uid == expected['uid'] and int(row[19]) == expected['start_ticks'] and row[0] != 'Z'
    except FileNotFoundError:
        return False


def resource_status():
    p = protocol()
    if not Path('/proc/meminfo').exists():
        return dict(admitted=False, blockers=['SERVER_ONLY_NO_LOCAL_GPU_ADMISSION'], UTC=now())
    blockers = []
    coco = Path(p['existing_COCO_output']) / 'pipeline_state.json'
    if coco.exists():
        state = load(coco)
        if state.get('controller') and process_live(state['controller']):
            blockers.append('EXISTING_COCO_SERIAL_CONTROLLER_LIVE')
        if state.get('active_child') and process_live(state['active_child']):
            blockers.append('EXISTING_COCO_SCIENTIFIC_CHILD_LIVE')
    mem = int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines()
                   if v.startswith('MemAvailable:'))) / 2**20
    owned = Path(p['remote_owned_root'])
    disk = shutil.disk_usage(owned if owned.exists() else owned.parent).free / 2**30
    def ticks():
        return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    io = []
    for _ in range(3):
        a = ticks(); time.sleep(1); b = ticks()
        d = [y - x for x, y in zip(a, b)]
        io.append(100 * d[4] / max(1, sum(d)))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    free = {r.split(',')[0].strip(): int(r.split(',')[1]) for r in gpu.splitlines()}['0']
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    owned_gpu = []
    for raw in apps.splitlines():
        if raw.strip():
            pid = int(raw.strip())
            try:
                if (Path('/proc') / str(pid)).stat().st_uid == os.getuid():
                    owned_gpu.append(pid)
            except FileNotFoundError:
                pass
    if owned_gpu:
        blockers.append('OWN_GPU_JOB_ALREADY_ACTIVE')
    if mem < 32: blockers.append('RAM_AVAILABLE_BELOW_32GiB')
    if disk < 150: blockers.append('DISK_FREE_BELOW_150GiB')
    if free < 24000: blockers.append('GPU_FREE_BELOW_24000MiB')
    if all(v >= 20 for v in io): blockers.append('SUSTAINED_IOWAIT_THREE_WINDOWS')
    return dict(admitted=not blockers, blockers=blockers, RAM_available_GiB=mem,
                disk_free_GiB=disk, GPU_free_MiB=free, IOwait_percent=io,
                own_GPU_pids=owned_gpu, other_user_processes_not_modified=True,
                CPU_load_and_GPU_utilization_are_advisory=True, UTC=now())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    from .worker import gate
    p, source = protocol(), commit()
    gate(args.preflight, source)
    resources = resource_status()
    if not args.run or not resources['admitted']:
        print(dict(state='PENDING' if not resources['admitted'] else 'READY_NOT_LAUNCHED', resources=resources))
        return
    owned = Path(p['remote_owned_root']).resolve()
    if not args.out.resolve().is_relative_to(owned):
        raise ValueError('AT output escaped its own independent root')
    args.out.mkdir(parents=True, exist_ok=False)
    lock = owned / 'AT_SERIAL_CONTROLLER_LOCK.json'
    write_once(lock, dict(pid=os.getpid(), run_root=str(args.out), source_commit=source, UTC=now()))
    state = dict(state='RUNNING', source_commit=source, stages={}, current_run=None, UTC=now())
    try:
        for run_id in p['order']:
            resources = resource_status()
            while not resources['admitted']:
                state.update(state='PENDING_RESOURCES', resources=resources)
                atomic(args.out / 'pipeline_state.json', state)
                time.sleep(300)  # this same admitted serial controller; no second waiter
                resources = resource_status()
            state['state'] = 'RUNNING'
            out = args.out / run_id
            state['current_run'] = run_id
            atomic(args.out / 'pipeline_state.json', state)
            with (args.out / (run_id + '.console.log')).open('xb') as log:
                code = subprocess.run([sys.executable, '-B', '-m', 'robustness_s4_at.worker',
                    '--formal', '--run-id', run_id, '--device', '0', '--out', str(out),
                    '--source-commit', source, '--preflight', str(args.preflight.resolve())],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
            accepted = out / 'accepted.json'
            if code or not accepted.exists() or load(accepted).get('state') != 'ACCEPTED':
                raise RuntimeError('AT arm failed; no automatic retry: ' + run_id)
            state['stages'][run_id] = dict(state='ACCEPTED', receipt_sha256=sha(accepted))
        from .report import aggregate
        aggregate(args.out)
        state.update(state='COMPLETED_AND_ACCEPTED', current_run=None)
    except BaseException as exc:
        state.update(state='FAILED', error=repr(exc))
        raise
    finally:
        state['UTC'] = now()
        atomic(args.out / 'pipeline_state.json', state)
        # Retain the lock and its terminal state; never delete or reuse silently.
        atomic(owned / 'AT_SERIAL_CONTROLLER_TERMINAL.json', state)


if __name__ == '__main__':
    main()
