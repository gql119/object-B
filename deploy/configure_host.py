#!/usr/bin/env python3
"""Map configurable mounted data into the exact accepted legacy paths.

This preserves source/label identity fields in materialization_manifest.json and
leaves PROTOCOL.json byte-exact. Run as the cloud login user with permission to
create the two symlinks; this script never uses sudo and never overwrites paths.
"""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def verify_source_closure(root):
    paths=sorted((root/'robustness_s4_at').glob('*.py'))
    paths.extend([root/'robustness_s4_at/configs/PROTOCOL.json',root/'configs/voc_yolov8n_20cls.yaml'])
    for path in paths:
        rel=path.relative_to(root).as_posix()
        try: expected=subprocess.check_output(['git','show','HEAD:'+rel],cwd=root,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError: raise SystemExit(f'source file is not present in pinned commit: {rel}')
        if hashlib.sha256(expected).hexdigest()!=sha(path): raise SystemExit(f'scientific source differs from pinned commit: {rel}')
def validate_link(dest,target):
    dest=Path(dest); target=Path(target).resolve()
    if dest.exists() or dest.is_symlink():
        if not dest.is_symlink() or dest.resolve()!=target: raise SystemExit(f'refusing to replace existing path: {dest}')
def link(dest,target):
    dest=Path(dest); target=Path(target).resolve()
    validate_link(dest,target)
    if dest.is_symlink(): return
    dest.parent.mkdir(parents=True,exist_ok=True)
    os.symlink(target,dest,target_is_directory=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--clean-root',type=Path,required=True); ap.add_argument('--s4-parent-root',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args(); root=a.source_root.resolve()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    expected=os.environ.get('S4_AT_EXPECTED_SOURCE_COMMIT')
    if not expected or expected=='SET_FROM_PUBLISHED_CLOUD_RELEASE':
        raise SystemExit('S4_AT_EXPECTED_SOURCE_COMMIT must be set from the published cloud release')
    if head!=expected: raise SystemExit(f'expected published cloud release {expected}, found {head}')
    verify_source_closure(root)
    proto=root/'robustness_s4_at/configs/PROTOCOL.json'; value=json.loads(proto.read_text(encoding='utf-8-sig'))
    for rel in ('images/train','labels/train','images/val','labels/val'):
        if not (a.clean_root/rel).is_dir(): raise SystemExit(f'missing clean input {rel}')
    for rel in ('dataset/images/train','dataset/labels/train','materialization_manifest.json','dataset_integrity.json',
                'noise/checkpoints/step_009144_epoch_24_cursor_00000.pt'):
        if not (a.s4_parent_root/rel).is_file() and not (a.s4_parent_root/rel).is_dir(): raise SystemExit(f'missing accepted S4 dependency {rel}')
    for rel,digest in value['reused_artifacts'].items():
        if sha(a.s4_parent_root/rel)!=digest: raise SystemExit(f'accepted provenance SHA mismatch: {rel}')
    if a.receipt.exists(): raise SystemExit(f'refusing to overwrite receipt {a.receipt}')
    validate_link(value['clean_root'],a.clean_root)
    validate_link(value['parent_output'],a.s4_parent_root)
    link(value['clean_root'],a.clean_root)
    link(value['parent_output'],a.s4_parent_root)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps({'state':'PATH_ALIASES_READY','source_commit':head,
        'protocol_sha256':sha(proto),'clean_alias':value['clean_root'],'clean_target':str(a.clean_root.resolve()),
        'parent_alias':value['parent_output'],'parent_target':str(a.s4_parent_root.resolve()),
        'poison_alias':value['poison_root'],'poison_target':str((a.s4_parent_root/'dataset').resolve()),
        'protocol_modified':False,'reused_artifacts_verified':value['reused_artifacts']},indent=2)+'\n',encoding='utf-8')
    print('Legacy path aliases ready; PROTOCOL.json is unchanged.')
if __name__=='__main__': main()
