#!/usr/bin/env python3
"""Read-only, full SHA256/identity audit of the accepted VOC20 trees.

This is not run by the local package builder: it reads several GB of source data.
It never changes/re-encodes images, labels, or provenance.
"""
import argparse, hashlib, json
from pathlib import Path

EXTS={'.jpg','.jpeg','.png','.bmp'}
EXPECTED={'clean_train':16551,'s4_train':16551,'clean_val':4952}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def images(folder):
    folder=Path(folder)
    if not folder.is_dir(): raise ValueError(f'missing {folder}')
    out={}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in EXTS: continue
        if p.is_symlink() or not p.is_file() or p.stem in out: raise ValueError(f'unsafe/duplicate image {p}')
        out[p.stem]=p
    return out
def labels(folder):
    folder=Path(folder)
    if not folder.is_dir(): raise ValueError(f'missing {folder}')
    out={p.stem:p for p in sorted(folder.glob('*.txt'))}
    if any(p.is_symlink() or not p.is_file() for p in out.values()): raise ValueError(f'unsafe label in {folder}')
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clean-root',type=Path,required=True)
    ap.add_argument('--s4-root',type=Path,required=True)
    ap.add_argument('--parent-output',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    trees={
      'clean_train':(images(a.clean_root/'images/train'),labels(a.clean_root/'labels/train')),
      's4_train':(images(a.s4_root/'images/train'),labels(a.s4_root/'labels/train')),
      'clean_val':(images(a.clean_root/'images/val'),labels(a.clean_root/'labels/val'))}
    for name,(im,la) in trees.items():
        if len(im)!=EXPECTED[name] or set(im)!=set(la): raise ValueError(f'{name} count/stem mismatch')
    clean,s4,val=trees['clean_train'][0],trees['s4_train'][0],trees['clean_val'][0]
    if set(clean)!=set(s4) or set(clean)&set(val): raise ValueError('paired train or clean val split mismatch')
    from PIL import Image
    rows={}
    for name,(im,la) in trees.items():
        items=[]
        for stem in sorted(im):
            p=im[stem]
            try:
                with Image.open(p) as image: image.verify()
                with Image.open(p) as image: width,height,fmt=image.width,image.height,image.format
            except Exception as exc: raise ValueError(f'damaged image {p}: {exc}') from exc
            raw=la[stem].read_bytes(); classes=[]
            for n,line in enumerate(raw.decode('utf-8').splitlines(),1):
                if not line.strip(): continue
                fields=line.split()
                if len(fields)!=5: raise ValueError(f'{la[stem]}:{n}: invalid YOLO row')
                cls,*xywh=map(float,fields)
                if cls!=int(cls) or not 0<=int(cls)<20 or any(x<0 or x>1 for x in xywh): raise ValueError(f'{la[stem]}:{n}: invalid VOC20 box')
                classes.append(int(cls))
            if name=='s4_train' and raw!=trees['clean_train'][1][stem].read_bytes(): raise ValueError(f'S4 changed GT bytes: {stem}')
            root=a.s4_root if name=='s4_train' else a.clean_root
            items.append({'id':stem,'image':{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'format':fmt,'width':width,'height':height},
                          'label':{'path':la[stem].relative_to(root).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'classes':classes}})
        rows[name]=items
    accepted=json.loads((a.parent_output/'materialization_manifest.json').read_text(encoding='utf-8-sig'))
    accepted_by_id={row['id']:row for row in accepted}
    clean_images,clean_labels=trees['clean_train']; s4_images,s4_labels=trees['s4_train']
    if len(accepted)!=EXPECTED['s4_train'] or len(accepted_by_id)!=len(accepted) or set(accepted_by_id)!=set(clean_images):
        raise ValueError('accepted materialization IDs do not match train image IDs')
    clean_rows={row['id']:row for row in rows['clean_train']}; s4_rows={row['id']:row for row in rows['s4_train']}
    for stem,record in accepted_by_id.items():
        if Path(record['source']).resolve()!=clean_images[stem].resolve() or Path(record['output']).resolve()!=s4_images[stem].resolve():
            raise ValueError(f'accepted source/output path identity mismatch: {stem}')
        if record.get('label_sha256')!=clean_rows[stem]['label']['sha256'] or record.get('output_sha256')!=s4_rows[stem]['image']['sha256']:
            raise ValueError(f'accepted image/label hash mismatch: {stem}')
    required={
      'materialization_manifest.json':'194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36',
      'dataset_integrity.json':'3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82',
      'noise/checkpoints/step_009144_epoch_24_cursor_00000.pt':'4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5'}
    provenance={}
    for rel,want in required.items():
        p=a.parent_output/rel
        if not p.is_file() or p.is_symlink() or sha(p)!=want: raise ValueError(f'accepted provenance mismatch: {rel}')
        provenance[rel]={'bytes':p.stat().st_size,'sha256':want}
    payload={'schema':'s4.at.cloud-data-manifest.v1','state':'FULL_SOURCE_HASH_SCAN_PASS','counts':EXPECTED,
             'roots':{'clean_train':'CLEAN_ROOT','s4_train':'S4_PARENT_ROOT/dataset','clean_val':'CLEAN_ROOT'},
             'person_class_id':14,'nc':20,'provenance':provenance,'trees':rows,
             'scope_note':'Transfer identity/format/label check only. Accepted Original S4 8/4/0 evidence remains the scientific budget authority.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
if __name__=='__main__': main()
