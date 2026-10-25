"""Safe RRCTL paths and exact, idempotent uploads for immutable phase releases.

RRCTL anchors must sit outside every lifecycle root. Formal phase execution,
however, reads ``source_release.json`` from its immutable source root. Keep a
separate anchor while the candidate is being smoke-tested, then copy those
exact bytes into the release root when registering the reviewed phase.
"""
import json
from pathlib import Path, PurePosixPath

from .phase_contract import OWNED
from .runtime import sha, write_once

PYTHON = '/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
REMOTE_OWNED = PurePosixPath(OWNED.as_posix())
REMOTE_RELEASES = REMOTE_OWNED / 'releases'
REMOTE_OUTPUT = REMOTE_OWNED / 'outputs' / 's4_coco_experiment_a_seed0_20261025_v3'

REMOTE_INSPECT = r'''import hashlib,json,os,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve(); out={}
for name in sys.argv[2:]:
 p=Path(name)
 if not p.is_absolute() or p.is_symlink(): raise ValueError('unsafe remote receipt path')
 parent=p.parent.resolve()
 if not parent.is_relative_to(base): raise ValueError('remote path escaped owned root')
 if not p.exists(): out[name]=None; continue
 if not p.is_file(): raise ValueError('remote receipt is not a regular file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''): h.update(b)
 out[name]=h.hexdigest()
print(json.dumps(out,sort_keys=True))'''


def anchor_path(tag):
    if not tag or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for ch in tag):
        raise ValueError('Unsafe candidate tag for release anchor')
    return (REMOTE_RELEASES / '_anchors' / tag / 'source_release.json').as_posix()


def validate_anchor_outside_lifecycle(remote_anchor, remote):
    anchor = PurePosixPath(remote_anchor)
    if not anchor.is_relative_to(REMOTE_RELEASES) or anchor == REMOTE_RELEASES:
        raise ValueError('RRCTL release anchor must stay within the owned releases tree')
    roots = [PurePosixPath(remote[key]) for key in ('repo_root', 'output_root', 'stage_root', 'control_root')]
    if any(anchor == root or anchor.is_relative_to(root) for root in roots):
        raise ValueError('RRCTL release anchor overlaps a lifecycle root')
    return remote_anchor


def inspect_remote(transport, paths):
    paths = list(dict.fromkeys(str(path) for path in paths))
    if any(not PurePosixPath(path).is_absolute() or not PurePosixPath(path).is_relative_to(REMOTE_OWNED)
           for path in paths):
        raise ValueError('RRCTL inspection is restricted to the owned COCO root')
    result = transport.run([PYTHON, '-B', '-c', REMOTE_INSPECT, REMOTE_OWNED.as_posix(), *paths])
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2500:])
    values = json.loads(result.stdout)
    if set(values) != set(paths):
        raise ValueError('Remote exact-readback omitted a requested path')
    return values


def ensure_remote_exact(transport, local_path, remote_path):
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    digest = sha(local_path)
    current = inspect_remote(transport, [remote_path])[remote_path]
    if current not in (None, digest):
        raise ValueError('Conflicting remote immutable file; refusing overwrite: ' + remote_path)
    if current is None:
        transport.upload(local_path, remote_path, digest)
    if inspect_remote(transport, [remote_path])[remote_path] != digest:
        raise ValueError('Remote immutable upload failed exact SHA-256 readback: ' + remote_path)
    return digest


def close_and_register(transport, *, tag, phase, module, spec, release_path, smoke_path,
                       review_path, previous_stage_receipt_sha256, closure_extra=None):
    """Upload exact release/review/closure/READY bytes; READY is always last."""
    spec = spec if isinstance(spec, dict) else json.loads(Path(spec).read_text(encoding='utf-8'))
    release_path = Path(release_path)
    smoke_path = Path(smoke_path)
    review_path = Path(review_path)
    release = json.loads(release_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if (release.get('phase') != phase or release.get('source_commit') != spec['source']['commit']
            or release.get('workflow_run_id') != 's4_coco_experiment_a_seed0_20261025_v3'):
        raise ValueError('Release identity differs from the exact candidate')
    if (review.get('phase') != phase or review.get('source_commit') != release['source_commit']
            or review.get('release_sha256') != sha(release_path)
            or review.get('phase_config_sha256') != sha(Path(spec['source']['repo_root']) / release['phase_config_path'])
            or review.get('smoke_sha256') != sha(smoke_path)
            or review.get('reviewer_count') != 1
            or review.get('reviewer_independence') is not True
            or review.get('pre_run_result') != 'pass'
            or review.get('scientifically_correct') is not True
            or review.get('allow_run') is not True
            or review.get('scientific_verdict') != 'allow_run'
            or review.get('blockers') not in ([], None)):
        raise ValueError('A single independent current-candidate allow_run review is required')
    if not smoke_path.is_file() or not review_path.is_file():
        raise FileNotFoundError('Retained smoke and review files are required')

    remote_root = spec['remote']['repo_root']
    if remote_root != (REMOTE_RELEASES / tag).as_posix():
        raise ValueError('Candidate source root is not the unique owned release directory')
    release_remote = remote_root + '/source_release.json'
    smoke_remote = spec['remote']['output_root'] + '/result/smoke_summary.json'
    receipt_root = REMOTE_OUTPUT / 'receipts'
    review_remote = (receipt_root / (phase + '_SCIENTIFIC_REVIEW.json')).as_posix()
    closure_remote = (receipt_root / (phase + '_SCIENTIFIC_CLOSURE.json')).as_posix()
    ready_remote = (receipt_root / (phase + '_READY.json')).as_posix()

    cfg_sha = sha(Path(spec['source']['repo_root']) / release['phase_config_path'])
    closure_core = dict(phase=phase, source_commit=release['source_commit'], release_sha256=sha(release_path),
                        phase_config_sha256=cfg_sha, smoke_path=smoke_remote, smoke_sha256=sha(smoke_path),
                        review_path=review_remote, review_sha256=sha(review_path), reviewer_count=1,
                        reviewer_independence=True, pre_run_result='pass', scientific_verdict='allow_run',
                        scientifically_correct=True, allow_run=True, blockers=[],
                        teacher_checkpoint_sha256=release['teacher_dependency']['checkpoint_sha256'])
    closure_core.update(closure_extra or {})
    closure_path = Path(release_path).parent / 'scientific_closure.json'
    if closure_path.exists():
        closure = json.loads(closure_path.read_text(encoding='utf-8'))
        if any(closure.get(k) != v for k, v in closure_core.items()) or not closure.get('closed_UTC'):
            raise ValueError('Retained local closure conflicts with this exact review')
    else:
        from .runtime import now
        closure = dict(closure_core, closed_UTC=now())
        write_once(closure_path, closure)

    ready_core = dict(phase=phase, workflow_run_id=release['workflow_run_id'], pre_run_result='pass',
                      source_root=remote_root, release_sha256=sha(release_path),
                      release_relative='source_release.json', module=module,
                      previous_stage_receipt_sha256=previous_stage_receipt_sha256,
                      scientific_closure_path=closure_remote, scientific_closure_sha256=sha(closure_path),
                      smoke_path=smoke_remote, smoke_sha256=sha(smoke_path))
    registration_path = Path(release_path).parent / 'phase_registration.json'
    if registration_path.exists():
        ready = json.loads(registration_path.read_text(encoding='utf-8'))
        if any(ready.get(k) != v for k, v in ready_core.items()) or not ready.get('registered_UTC'):
            raise ValueError('Retained READY differs from the exact reviewed release')
    else:
        from .runtime import now
        ready = dict(ready_core, registered_UTC=now())
        write_once(registration_path, ready)

    if inspect_remote(transport, [smoke_remote])[smoke_remote] != sha(smoke_path):
        raise ValueError('Retained RRCTL smoke differs from the pulled summary')
    ensure_remote_exact(transport, release_path, release_remote)
    ensure_remote_exact(transport, review_path, review_remote)
    ensure_remote_exact(transport, closure_path, closure_remote)
    ensure_remote_exact(transport, registration_path, ready_remote)
    return dict(phase=phase, reviewer_count=1, registered_existing_controller=True,
                READY_uploaded_and_readback=True, serial_controller_owns_formal_start=True,
                no_new_controller=True, release_sha256=sha(release_path), ready_sha256=sha(registration_path))
