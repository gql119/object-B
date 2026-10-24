"""Enter an ordinary independent user scope, retain identity, then exec the job.

No priority, memory limit, oom_score_adj, ManagedOOM or protection override.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys

from .runtime import now, write_once


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    p = Path('/proc')/str(pid)
    return dict(pid=pid, start_ticks=int((p/'stat').read_text().rsplit(')', 1)[1].split()[19]), uid=p.stat().st_uid)


def cgroup(pid=None):
    pid = os.getpid() if pid is None else pid
    rel = (Path('/proc')/str(pid)/'cgroup').read_text().strip().split('0::', 1)[1]
    root = Path('/sys/fs/cgroup')/rel.lstrip('/')
    return dict(path=rel, **{n:(root/n).read_text().strip() for n in
                ('memory.current','memory.peak','memory.events','memory.pressure') if (root/n).is_file()})


def scope_command(job, argv, cwd):
    unit = 's4coco-job-'+hashlib.sha256(str(Path(job).resolve()).encode()).hexdigest()[:24]
    return unit, ['systemd-run','--user','--scope','--quiet','--unit='+unit,
                  sys.executable,'-B','-m','coco_s4.process','--job',str(job),'--cwd',str(cwd),'--',*argv]


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--cwd',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd or os.getuid()!=1014:
        raise ValueError('Explicit owned scientific child required')
    write_once(a.job/'child.json',dict(**identity(),argv=cmd,cwd=str(a.cwd),UTC=now(),memory_scope=cgroup()))
    os.chdir(a.cwd)
    os.execvpe(cmd[0],cmd,os.environ.copy())


if __name__=='__main__':
    main()
