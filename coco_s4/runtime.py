"""Small immutable release and data bindings for the serial COCO workflow."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023')
PARENT = 'b1e29510e7b1ed5fa0fe361a27132d8790e65c14'
VICTIM_PARENT = 'ab626917f20917e58cf30033ac99df00938cb561'
DATA_FILES = {
    'metadata/instances_minitrain2017.json': '5682389c3bb0a356912771811f7836659cc6de5a588413752922ccb94a6df1a8',
    'metadata/instances_val2017.json': 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f',
    'metadata/TRAIN25K_KAGGLE_IMAGE_MANIFEST_20261024.jsonl': 'a8bd9bd336685bba1573d0401ad686f76e156daf35ea9867499a34c2fb31a343',
    'metadata/VAL2017_IMAGE_MANIFEST_20261025.jsonl': '6fef2d07ef94fbc190519f44959eacdca0139d551514c671bf42843d33b18f67',
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def commit(root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe release path')
    return p


def verify_release(manifest_path, *, native=True):
    """Hash the committed computation closure, not mutable progress/report files."""
    m = load(manifest_path)
    scope=load(ROOT/'coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json')
    if not (m.get('validation_only') is True and m.get('repair_authorization_sha256')==scope['authorization_sha256']):
        assert_formal_launch_authorized()
    if m['schema'] != 's4.coco.release.v1' or m['scientific_parent'] != PARENT or m['victim_parent'] != VICTIM_PARENT:
        raise ValueError('Original S4 scientific identity differs')
    if m['source_commit'] != commit():
        raise ValueError('Runtime commit differs from bound candidate')
    for rel, expected in m['files'].items():
        p = ROOT / safe_relative(rel)
        if not p.resolve().is_relative_to(ROOT) or sha(p) != expected:
            raise ValueError('Computation source drift: ' + rel)
    if m['data_files'] != DATA_FILES or m['data_root'] != str(DATA_ROOT):
        raise ValueError('Data identity changed')
    if native:
        if os.getuid() != 1014:
            raise ValueError('Wrong remote owner')
        u = Path(importlib.util.find_spec('ultralytics').origin).parent
        for rel, expected in m['native_sources'].items():
            if sha(u / safe_relative(rel)) != expected:
                raise ValueError('Native implementation drift: ' + rel)
        for package, version in m['versions'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError('Environment drift: ' + package)
        for rel, expected in DATA_FILES.items():
            if sha(DATA_ROOT / rel) != expected:
                raise ValueError('Accepted data receipt/metadata drift: ' + rel)
    return m


def assert_formal_launch_authorized():
    """Current human conditional approval plus actual closed repair gates; never old v2."""
    scope=load(ROOT/'coco_s4/configs/EMERGENCY_REPAIR_SCOPE.json')
    if scope.get('formal_relaunch_allowed') is not True:
        raise ValueError('EMERGENCY_REPAIR_ONLY: explicit human formal-relaunch authorization required')
    from .phase_contract import WORKFLOW,INVALIDATED_WORKFLOW
    if scope.get('workflow_run_id')!=WORKFLOW or WORKFLOW==INVALIDATED_WORKFLOW:
        raise ValueError('Formal authorization is not bound to the unique repaired workflow')
    if sha(ROOT/safe_relative(scope['human_relaunch_authorization_path']))!=scope['human_relaunch_authorization_sha256']:
        raise ValueError('Direct human relaunch authorization changed')
    ready=load(ROOT/safe_relative(scope['actual_readiness_path']))
    if (ready.get('state')!='ALL_REPAIR_VALIDATIONS_PASSED' or ready.get('native_CPU16')!='passed'
            or ready.get('native_CUDA16')!='passed' or ready.get('extra_exact_export_device_observer')!='passed'
            or ready.get('read_only_main_audits')!='passed' or ready.get('consumer_simulation')!='passed'):
        raise ValueError('Actual repair validation gates remain open')


def require_review(path, release_path, smoke_path):
    r = load(path)
    smoke = load(smoke_path)
    expected = dict(source_commit=commit(), release_sha256=sha(release_path),
                    smoke_sha256=sha(smoke_path), pre_run_result='pass', reviewer_count=1,
                    phase='teacher')
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Missing exact current-candidate scientific closure')
    if (smoke.get('state') != 'passed' or smoke.get('source_commit') != commit()
            or not smoke.get('finite_loss') or not smoke.get('production_entrypoint_reached')):
        raise ValueError('Current Teacher GPU evidence required')
    return r
