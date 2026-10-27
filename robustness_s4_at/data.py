"""Read-only accepted VOC inputs; native caches live in a new AT data view."""
from pathlib import Path
import numpy as np
import cv2
import yaml
from .common import load, protocol, sha, write_once

VOC_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
             'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
             'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def image_index(folder):
    result = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        if p.stem in result or not p.is_file():
            raise ValueError('Duplicate or missing image stem')
        result[p.stem] = p
    return result


def audit_accepted_inputs(p):
    parent, clean, poison = (Path(p[k]) for k in ('parent_output', 'clean_root', 'poison_root'))
    for rel, digest in p['reused_artifacts'].items():
        if sha(parent / rel) != digest:
            raise ValueError('Accepted Original S4 dependency SHA mismatch: ' + rel)
    train, val, protected = (image_index(path) for path in
                             (clean / 'images/train', clean / 'images/val', poison / 'images/train'))
    if len(train) != 16551 or len(val) != 4952 or set(train) != set(protected) or set(train) & set(val):
        raise ValueError('Original VOC split identity/count mismatch')
    accepted = load(parent / 'materialization_manifest.json')
    manifest = {row['id']: row for row in accepted}
    if len(accepted) != 16551 or set(manifest) != set(train):
        raise ValueError('Accepted materialization manifest ID mismatch')
    for split, images in (('train', train), ('val', val)):
        if {v.stem for v in (clean / 'labels' / split).glob('*.txt')} != set(images):
            raise ValueError('Clean VOC image/label ID mismatch')
    records, samples = [], []
    selected = {'person': None, 'no_person': None, 'cooccurrence': None}
    for stem in train:
        label = clean / 'labels/train' / (stem + '.txt')
        other = poison / 'labels/train' / label.name
        if not label.is_file() or not other.is_file() or label.read_bytes() != other.read_bytes():
            raise ValueError('Static S4 label differs: ' + stem)
        row = manifest[stem]
        if sha(label) != row['label_sha256'] or Path(row['source']).resolve() != train[stem].resolve() \
                or Path(row['output']).resolve() != protected[stem].resolve():
            raise ValueError('Accepted materialization source/label identity drift')
        entries = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        if any(len(row) != 5 or row[0] != int(row[0]) or not 0 <= row[0] < 20 for row in entries):
            raise ValueError('Malformed VOC20 label: ' + stem)
        classes = [int(row[0]) for row in entries]
        kind = 'person' if 14 in classes else 'no_person'
        selected[kind] = selected[kind] or stem
        if 14 in classes and any(k != 14 for k in classes):
            selected['cooccurrence'] = selected['cooccurrence'] or stem
        records.append(dict(stem=stem, sha256=sha(label), classes=classes))
    # Retained full acceptance covers every image; this is explicitly a sample.
    from ue_framework.methods.legacy_kproto_ret import _rectangle
    for stem in sorted(set(v for v in selected.values() if v is not None)):
        x, y = cv2.imread(str(train[stem])), cv2.imread(str(protected[stem]))
        if x is None or y is None or x.shape != y.shape:
            raise ValueError('Invalid accepted static image pair')
        person = np.zeros(x.shape[:2], dtype=bool)
        nt = np.zeros_like(person)
        for line in (clean / 'labels/train' / (stem + '.txt')).read_text().splitlines():
            row = list(map(float, line.split()))
            box = _rectangle(x.shape[:2], row[1:])[0, 0].numpy().astype(bool)
            if int(row[0]) == 14:
                person |= box
            else:
                nt |= box
        delta = np.abs(y.astype(np.int16) - x.astype(np.int16)).max(2)
        if np.any(delta[~person] != 0) or np.any(delta[person & nt] > 4) or np.any(delta[person & ~nt] > 8):
            raise ValueError('Original S4 decoded 8/4/0 sample failed: ' + stem)
        samples.append(dict(stem=stem, clean_sha256=sha(train[stem]), poison_sha256=sha(protected[stem]),
                            max_exclusive=int(delta[person & ~nt].max(initial=0)),
                            max_overlap=int(delta[person & nt].max(initial=0)),
                            max_outside=int(delta[~person].max(initial=0))))
        if sha(protected[stem]) != manifest[stem]['output_sha256']:
            raise ValueError('Accepted sample image byte identity drift')
    return dict(train=train, val=val, poison=protected, label_manifest=records, samples=samples)


def data_view(out, arm, *, fixture_root=None):
    p, out = protocol(), Path(out)
    if fixture_root is not None:
        fixture_root = Path(fixture_root)
        train = image_index(fixture_root / 'images/train')
        val = train  # train fixture only; all validators must be disabled.
        source_labels = clean_labels = fixture_root / 'labels/train'
        audit = dict(state='EXPLICIT_TRAIN_ONLY_FIXTURE', evaluation_allowed=False,
                     train_fixture_count=len(train), original_counts_preserved=False)
        path = out / 'data.yaml'
        path.write_text(yaml.safe_dump(dict(train=str((fixture_root / 'images/train').resolve()),
            val=str((fixture_root / 'images/train').resolve()),
            names=dict(enumerate(VOC_NAMES)), nc=20), sort_keys=False), encoding='utf-8')
        write_once(out / 'data_audit.json', audit)
        return path
    else:
        checked = audit_accepted_inputs(p)
        train = checked['train'] if arm == 'clean' else checked['poison']
        val = checked['val']
        source_labels = Path(p['clean_root'] if arm == 'clean' else p['poison_root']) / 'labels/train'
        clean_labels = Path(p['clean_root']) / 'labels/val'
        audit = {k: v for k, v in checked.items() if k not in ('train', 'val', 'poison')}
        audit.update(train_count=len(train), val_count=len(val), validation_clean=True,
                     all_training_labels_byte_unchanged=True, original_acceptance=p['reused_acceptance'])
    view = out / 'data'
    if view.exists():
        raise FileExistsError('Fresh AT data view collision')
    for split, images, labels in (('train', train, source_labels), ('val', val, clean_labels)):
        (view / 'images' / split).mkdir(parents=True)
        (view / 'labels' / split).mkdir(parents=True)
        for stem, image in images.items():
            (view / 'images' / split / image.name).symlink_to(image.resolve())
            (view / 'labels' / split / (stem + '.txt')).symlink_to((labels / (stem + '.txt')).resolve())
    path = out / 'data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(view.resolve()), train='images/train', val='images/val',
                                        names=dict(enumerate(VOC_NAMES)), nc=20), sort_keys=False), encoding='utf-8')
    write_once(out / 'data_audit.json', audit)
    return path
