"""Read-only dataset audit / parent-evidence binder. Does not launch training."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ue_framework.data_utils import read_yolo_annotations,label_path_for_image
from ue_framework.methods.csb_budget import digest
from ue_framework.methods.counterfactual_parent import select_parent


def train_manifest(train_images,train_labels,val_images,target=14):
    suffixes={'.jpg','.jpeg','.png'}
    paths=sorted(p.resolve() for p in Path(train_images).iterdir() if p.suffix.lower() in suffixes)
    validation=sorted(p.resolve() for p in Path(val_images).iterdir() if p.suffix.lower() in suffixes)
    if not paths or not validation:
        raise ValueError('Both real train and clean validation identity lists required')
    if ({p.stem for p in paths}&{p.stem for p in validation}
            or len({p.stem for p in paths}) != len(paths)
            or len({p.stem for p in validation}) != len(validation)):
        raise ValueError('Train/val stem overlap or duplicate image identity')
    rows=[]
    for path in paths:
        label=Path(label_path_for_image(path,str(train_labels))).resolve()
        if not label.is_file():
            raise ValueError('Missing train label: '+str(label))
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            fields=line.split()
            if len(fields)!=5:
                raise ValueError('Malformed YOLO train label: '+str(label))
            values=[float(x) for x in fields]
            if not all(math.isfinite(x) for x in values) or not values[0].is_integer():
                raise ValueError('Nonfinite/noninteger GT: '+str(label))
        anns=read_yolo_annotations(label)
        for a in anns:
            c=int(a['cls']); box=a['bbox']
            if not 0<=c<20 or box is None or len(box)!=4 or any(not 0<=v<=1 for v in box) or min(box[2:])<=0:
                raise ValueError('Invalid GT in '+str(label))
        n=sum(int(a['cls'])!=target for a in anns)
        if n:
            rows.append(dict(image_id=str(path),image_path=str(path),label_path=str(label),
                             image_sha256=digest(path),label_sha256=digest(label),non_target_gt_count=n))
    if not rows: raise ValueError('No non-target train queries')
    return dict(status='AUDITED_TRAIN_ONLY',split='train',target_class_id=target,
                full_train_image_ids=[str(p) for p in paths],validation_image_ids=[str(p) for p in validation],
                records=rows,selection_rule='all train images with valid NT GT; never validation AP',
                validation_usage='identity leakage audit only, no pixels/labels/loss/AP used')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    m=sub.add_parser('train-query-manifest')
    m.add_argument('--train-images',required=True); m.add_argument('--train-labels',required=True)
    m.add_argument('--val-images',required=True); m.add_argument('--output',required=True)
    p=sub.add_parser('select-parent')
    p.add_argument('--s0-evidence',required=True); p.add_argument('--s1-evidence',required=True)
    p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='train-query-manifest':
        value=train_manifest(args.train_images,args.train_labels,args.val_images)
    else:
        value=select_parent(json.loads(Path(args.s0_evidence).read_text(encoding='utf-8')),
                            json.loads(Path(args.s1_evidence).read_text(encoding='utf-8')))
    output=Path(args.output)
    # Exclusive create protects previous experiment evidence.
    with output.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(output=str(output),sha256=digest(output),status=value['status'])))


if __name__=='__main__': main()
