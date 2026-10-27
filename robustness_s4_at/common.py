"""Small retained records; no destructive filesystem or process operations."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'robustness_s4_at/configs/PROTOCOL.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def scientific_source():
    files = sorted((ROOT / 'robustness_s4_at').glob('*.py')) + [PROTOCOL]
    manifest, uncommitted = {}, []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        manifest[rel] = sha(path)
        result = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=ROOT, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != manifest[rel]:
            uncommitted.append(rel)
    return dict(commit=commit(), files=manifest, uncommitted_files=uncommitted)


def protocol():
    p = load(PROTOCOL)
    if p['scientific_parent'] != 'Original S4 = S0 Safe-Soft05 + TFSS-Utility':
        raise ValueError('Wrong scientific parent')
    if (p['target_id'], p['nc'], p['train_count'], p['val_count']) != (14, 20, 16551, 4952):
        raise ValueError('VOC20 dataset contract changed')
    a = p['train_args']
    expected = dict(epochs=200, batch=36, workers=16, imgsz=640, seed=0,
                    optimizer='SGD', amp=True, pretrained=False, cos_lr=True,
                    close_mosaic=100, patience=0, lr0=.01, lrf=.01,
                    momentum=.937, weight_decay=.0005)
    if any(a[k] != v for k, v in expected.items()):
        raise ValueError('Frozen native victim training protocol changed')
    if p['order'] != ['S4AT4_Clean_seed0', 'S4AT4_Poison_seed0',
                      'S4AT8_Clean_seed0', 'S4AT8_Poison_seed0']:
        raise ValueError('AT run order changed')
    for eps, step in ((4, 1), (8, 2)):
        if p['pgd'][str(eps)] != dict(epsilon_pixels=eps, step_pixels=step,
                                    steps=10, random_start=True):
            raise ValueError('PGD-10 protocol changed')
    return p


def run_config(run_id):
    p = protocol()
    if run_id not in p['order']:
        raise ValueError('Unknown independent AT RunID')
    eps = 4 if run_id.startswith('S4AT4_') else 8
    return dict(run_id=run_id, arm='clean' if '_Clean_' in run_id else 'poison',
                attack=p['pgd'][str(eps)], protocol_sha256=sha(PROTOCOL))


def native_identity(require_reference=True):
    import ultralytics
    root = Path(ultralytics.__file__).resolve().parent
    p = protocol()
    actual = {rel: sha(root / rel) for rel in p['native_sources']}
    if ultralytics.__version__ != '8.4.150':
        raise ValueError('Unreviewed Ultralytics version')
    different = [rel for rel, digest in actual.items() if digest != p['native_sources'][rel]]
    if require_reference and different:
        raise ValueError('Native scientific source drift: ' + ', '.join(different))
    return dict(version=ultralytics.__version__, sources=actual, differing_from_reference=different)
