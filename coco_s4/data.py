"""Audit the author JSON; deterministic COCO-ID mapping, never sample a split."""
from collections import Counter,defaultdict
import hashlib,json,math
from pathlib import Path

COCO_IDS=(1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90)

def category_mapping(data):
    categories=sorted(data['categories'],key=lambda x:x['id'])
    if tuple(x['id'] for x in categories)!=COCO_IDS or categories[0]['name']!='person':
        raise ValueError('Not the official COCO80 ID/category schema')
    if len({x['name'] for x in categories})!=80:raise ValueError('Duplicate class name')
    return [{'category_id':x['id'],'yolo_class_id':i,'name':x['name']} for i,x in enumerate(categories)]

def annotation_to_yolo(ann,image,mapping):
    x,y,w,h=map(float,ann['bbox']);iw,ih=image['width'],image['height']
    if not all(map(math.isfinite,(x,y,w,h))) or min(x,y)<0 or min(w,h)<=0 or x+w>iw+1e-6 or y+h>ih+1e-6:
        raise ValueError(f"Invalid original bbox annotation {ann['id']}; no clipping/drop fallback")
    if ann.get('iscrowd',0) or ann.get('ignore',0):return None
    cls=mapping[ann['category_id']]
    return (cls,(x+w/2)/iw,(y+h/2)/ih,w/iw,h/ih)

def audit_metadata(train,val,*,train_count=25000,val_count=5000):
    mapping=category_mapping(train)
    if category_mapping(val)!=mapping:raise ValueError('Train/val category schema differs')
    by_id={x['category_id']:x['yolo_class_id'] for x in mapping};splits={};ids={}
    labels={}
    for name,data,count in [('train',train,train_count),('val',val,val_count)]:
        images={x['id']:x for x in data['images']}
        if len(images)!=count or len(data['images'])!=count:raise ValueError('Image count/uniqueness differs: '+name)
        ids[name]=set(images);raw=Counter();eligible=Counter();sizes=Counter();crowd=Counter();ignored=Counter();anns=set();classes=defaultdict(set)
        labels[name]={i:[] for i in images}
        for i,im in images.items():
            if not isinstance(i,int) or min(im['width'],im['height'])<=0 or Path(im['file_name']).name!=f'{i:012d}.jpg':raise ValueError('Invalid COCO image identity/dimensions')
            if im.get('coco_url') and '/'+('train2017' if name=='train' else 'val2017')+'/' not in im['coco_url']:raise ValueError('Image outside declared COCO2017 split')
        for a in data['annotations']:
            if a['id'] in anns:raise ValueError('Duplicate annotation ID')
            anns.add(a['id'])
            if a['image_id'] not in images or a['category_id'] not in by_id:raise ValueError('Foreign image/category annotation')
            raw[a['category_id']]+=1;crowd[a['category_id']]+=int(bool(a.get('iscrowd',0)));ignored[a['category_id']]+=int(bool(a.get('ignore',0)))
            row=annotation_to_yolo(a,images[a['image_id']],by_id)
            if row is not None:
                labels[name][a['image_id']].append(row);eligible[a['category_id']]+=1;classes[a['image_id']].add(row[0])
                area=float(a['area']);sizes['small' if area<32**2 else 'medium' if area<96**2 else 'large']+=1
        if name=='train' and set(eligible)!=set(COCO_IDS):raise ValueError('A COCO class lacks eligible training GT')
        with_person=sum(0 in c for c in classes.values());cooccur=sum(0 in c and len(c)>1 for c in classes.values())
        splits[name]={'images':count,'raw_annotations':len(anns),'eligible_training_annotations':sum(eligible.values()),'crowd_annotations':sum(crowd.values()),'explicit_ignore_annotations':sum(ignored.values()),'per_class_raw':dict(raw),'per_class_eligible':dict(eligible),'size_counts_by_official_area':dict(sizes),'person_images':with_person,'person_with_NT_images':cooccur,'person_NT_cooccurrence_fraction':cooccur/with_person if with_person else None,'no_person_images':count-with_person,'empty_label_images':sum(not rows for rows in labels[name].values()),'image_ids_sha256':hashlib.sha256('\n'.join(map(str,sorted(images))).encode()).hexdigest()}
    if ids['train'] & ids['val']:raise ValueError('Train/validation image overlap')
    return {'state':'METADATA_VALIDATED_IMAGES_PENDING','mapping':mapping,'splits':splits,'disjoint':True,'crowd_ignore_policy':'exclude from YOLO training/feature statistics; retain original JSON including crowd for official COCOeval','label_policy':'all80 classes, normalized xywh, no bbox clipping, no label or image deletion; empty/no-Person images retained'},labels

def write_labels(labels,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for image_id,rows in labels.items():
        (root/f'{image_id:012d}.txt').write_text(''.join(' '.join([str(int(r[0])),*[format(x,'.17g') for x in r[1:]]])+'\n' for r in rows))

def audit_pixels(clean,poison,support):
    import numpy as np
    if clean.shape!=poison.shape or clean.dtype!=np.uint8 or poison.dtype!=np.uint8 or support.shape!=clean.shape[:2]:raise ValueError('Export identity/dtype mismatch')
    if not np.isin(support,[0,.5,1]).all():raise ValueError('Original S4 support must be8/4/0')
    d=np.abs(poison.astype(np.int16)-clean.astype(np.int16));budget=(support*8).astype(np.int16)
    if np.any(d>budget[...,None]):raise ValueError('Decoded PNG violates original8/4/0 budget')
    return {'max_bytes':int(d.max()),'outside_nonzero':int(np.count_nonzero(d[support==0])),'overlap_max_bytes':int(d[support==.5].max()) if np.any(support==.5) else 0}
