#!/usr/bin/env python3
"""Overlay only the 12 protocol-pinned Ultralytics files into a dedicated venv."""
import hashlib,json,os,shutil
from pathlib import Path
import ultralytics

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    if ultralytics.__version__!='8.4.150': raise SystemExit('Ultralytics direct pin mismatch')
    root=Path(os.environ['S4_AT_SOURCE_ROOT']).resolve()
    package=Path(os.environ['S4_AT_PACKAGE_ROOT']).resolve()
    owned=Path(os.environ['S4_AT_OWNED_ROOT']).resolve()
    proto=json.loads((root/'robustness_s4_at/configs/PROTOCOL.json').read_text(encoding='utf-8-sig'))
    native=Path(ultralytics.__file__).resolve().parent
    backup=owned/'environment_backup/ultralytics'
    rows=[]
    for rel,want in proto['native_sources'].items():
        src=package/'native_sources/ultralytics'/rel; dst=native/rel
        if not src.is_file() or sha(src)!=want: raise SystemExit(f'packaged pinned source mismatch: {rel}')
        actual=sha(dst)
        if actual==want:
            rows.append({'path':rel,'state':'already_exact','sha256':want}); continue
        saved=backup/rel
        if saved.exists(): raise SystemExit(f'refusing existing backup: {saved}')
        saved.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(dst,saved)
        shutil.copy2(src,dst)
        if sha(dst)!=want: raise SystemExit(f'installed source verification failed: {rel}')
        rows.append({'path':rel,'state':'installed_from_pinned_bundle','previous_sha256':actual,'sha256':want,'backup':str(saved)})
    receipt=owned/'environment_backup/native_source_install_receipt.json'
    if receipt.exists(): raise FileExistsError(receipt)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps({'state':'NATIVE_SOURCE_HASHES_VERIFIED','ultralytics_version':ultralytics.__version__,'files':rows},indent=2)+'\n',encoding='utf-8')
    print(receipt)
if __name__=='__main__': main()
