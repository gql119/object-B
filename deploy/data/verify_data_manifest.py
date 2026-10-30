#!/usr/bin/env python3
"""Rehash transferred files against the source-side relative-path manifest."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--clean-root',type=Path,required=True); ap.add_argument('--s4-parent-root',type=Path,required=True)
    a=ap.parse_args(); m=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    if m.get('schema')!='s4.at.cloud-data-manifest.v1' or m.get('state')!='FULL_SOURCE_HASH_SCAN_PASS': raise SystemExit('source manifest is not a completed full scan')
    roots={'clean_train':a.clean_root,'clean_val':a.clean_root,'s4_train':a.s4_parent_root/'dataset'}
    checked=0
    for tree,rows in m['trees'].items():
        root=roots[tree]
        for row in rows:
            for kind,base in (('image',root),('label',root)):
                item=row[kind]; path=base/item['path']
                if not path.is_file() or path.is_symlink() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                    raise SystemExit(f'transfer hash/size mismatch: {tree}/{row["id"]}/{kind}')
                checked+=1
    print(json.dumps({'state':'TRANSFER_HASH_PASS','files_checked':checked,'manifest':str(a.manifest)},indent=2))
if __name__=='__main__': main()
