"""Shared isolated probe contracts. No GPU/model/remote operation at import."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'execution/speed_audit_20261029'))
from robustness_s4_at.common import commit,protocol,sha,write_once

BASE_PGD_SHA='63bfa27086953566efb771e8bddf4aba5cc7363b8198dd3d73a28ba93df08578'


def admission(args,entry):
    p=protocol();gpu=args.device=='0'
    args.out=args.out.resolve();args.fixture_root=args.fixture_root.resolve()
    args.fixture_manifest=args.fixture_manifest.resolve()
    if commit()!=args.source_commit:raise ValueError('Exact source commit mismatch')
    if sha(ROOT/'robustness_s4_at/pgd.py')!=BASE_PGD_SHA:raise ValueError('Reference PGD drift')
    if args.out.exists():raise FileExistsError('Retained probe output collision')
    if not gpu:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        return dict(CPU_plumbing_only=True,CUDA_admission=False)
    paths=[Path(entry).resolve(),Path(__file__).resolve(),Path(__file__).parent/'gpu_telemetry.py',ROOT/'execution/speed_audit_20261029/candidate_scalar_readback.py']
    if hasattr(args,'batch'):paths.append(Path(__file__).parent/f'PROPOSED_TEST_PROTOCOL_B{args.batch}.json')
    for path in paths:
        blob=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=sha(path):raise ValueError('Uncommitted isolated GPU entry')
    if not args.out.is_relative_to(Path(p['remote_owned_root']).resolve()):raise ValueError('Unowned GPU probe output')
    if not args.fixture_manifest_sha256 or sha(args.fixture_manifest)!=args.fixture_manifest_sha256:
        raise ValueError('Fixture manifest is not hash-bound before GPU execution')
    from robustness_s4_at.queue import resource_status
    resources=resource_status()
    if not resources['admitted']:raise ValueError('Original resource/own-GPU gate refused: '+str(resources))
    # Benchmarks require no other GPU job, although formal resource admission
    # permits other UIDs. This adds an isolation condition; changes no rule.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    if any(s.strip() for s in apps.splitlines()):raise ValueError('Competing GPU job contaminates benchmark window')
    return resources


def fixture_view(args,destination,count=None):
    """Fresh cache namespace using retained train-only bytes; no full rescan."""
    from robustness_s4_at.data import image_index,data_view
    manifest=json.loads(args.fixture_manifest.read_text(encoding='utf8'))
    images=image_index(args.fixture_root/'images/train')
    gpu=args.device=='0'
    if gpu:
        rows={r['stem']:r for r in manifest['records'] if r['arm']==args.arm}
        if set(rows)!=set(images) or len(images)<684:raise ValueError('Retained GPU training pool identity mismatch')
    else:
        rows={r['image_id']:r for r in manifest['records']}
        if set(rows)!=set(images):raise ValueError('CPU fixture identity mismatch')
    selected=sorted(images)[:count] if count is not None else sorted(images)
    if count is not None and len(selected)!=count:raise ValueError('Insufficient distinct retained training fixtures')
    destination.mkdir(parents=True,exist_ok=False)
    records=[]
    for folder in ('images/train','labels/train'):(destination/folder).mkdir(parents=True)
    for stem in selected:
        image=images[stem];label=args.fixture_root/'labels/train'/(stem+'.txt');r=rows[stem]
        if sha(image)!=r['image_sha256'] or sha(label)!=r['label_sha256']:raise ValueError('Retained training bytes changed')
        for original,target in ((image,destination/'images/train'/image.name),(label,destination/'labels/train'/label.name)):
            if gpu:target.symlink_to(original.resolve())
            else:shutil.copyfile(original,target)
        records.append(dict(stem=stem,image_sha256=r['image_sha256'],label_sha256=r['label_sha256']))
    write_once(destination/'selected_identity.json',dict(train_only=True,arm=args.arm,records=records,
        parent_fixture_manifest_sha256=sha(args.fixture_manifest),selected_count=len(selected),full_original_census_repeated=False))
    data=data_view(destination,args.arm,fixture_root=destination)
    return data,records


def add_arguments(parser):
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--implementation',choices=('original','02','03'),required=True)
    parser.add_argument('--arm',choices=('clean','poison'),required=True)
    parser.add_argument('--epsilon',type=int,choices=(4,8),required=True)
    parser.add_argument('--device',choices=('0','cpu'),default='0')
    parser.add_argument('--fixture-root',type=Path,required=True)
    parser.add_argument('--fixture-manifest',type=Path,required=True)
    parser.add_argument('--fixture-manifest-sha256')
