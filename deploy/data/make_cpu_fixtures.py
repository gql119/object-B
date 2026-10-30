#!/usr/bin/env python3
"""Create a tiny symlink-only CPU fixture from accepted clean train data."""
import argparse
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clean-root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); images=a.clean_root/'images/train'; labels=a.clean_root/'labels/train'
    if a.out.exists(): raise FileExistsError(a.out)
    chosen={'person':None,'no_person':None,'cooccurrence':None}
    for p in sorted(labels.glob('*.txt')):
        classes={int(float(x.split()[0])) for x in p.read_text().splitlines() if x.strip()}
        if 14 in classes and chosen['person'] is None: chosen['person']=p
        if 14 not in classes and chosen['no_person'] is None: chosen['no_person']=p
        if 14 in classes and len(classes)>1 and chosen['cooccurrence'] is None: chosen['cooccurrence']=p
        if all(chosen.values()): break
    if not all(chosen.values()): raise ValueError('could not select person/no-person/co-occurrence fixtures')
    for folder in ('images/train','labels/train'): (a.out/folder).mkdir(parents=True,exist_ok=True)
    for p in chosen.values():
        image=next((images/(p.stem+ext) for ext in ('.jpg','.jpeg','.png','.bmp') if (images/(p.stem+ext)).is_file()),None)
        if image is None: raise FileNotFoundError('image missing for '+p.stem)
        (a.out/'images/train'/image.name).symlink_to(image.resolve())
        (a.out/'labels/train'/p.name).symlink_to(p.resolve())
    print('Created symlink-only fixture:',','.join(p.stem for p in chosen.values()))
if __name__=='__main__': main()
