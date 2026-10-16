"""Read train labels/headers only; deterministic seed0 stratified development set."""
import hashlib
import json
import math
from pathlib import Path
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(root, count=32, seed=0):
    root = Path(root)
    paths = {p.stem: p for p in (root / 'images/train').iterdir()
             if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
    records = []
    for path in sorted((root / 'labels/train').glob('*.txt')):
        rows = [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]
        if not any(int(row[0]) == 14 for row in rows):
            continue
        if any(len(row) != 5 or not all(math.isfinite(v) for v in row) for row in rows):
            raise ValueError('Invalid GT: ' + str(path))
        image = paths[path.stem]
        with Image.open(image) as im:
            w, h = im.size
        anns = []
        for i, (c, x, y, bw, bh) in enumerate(rows):
            if not (0 <= c < 20 and c == int(c) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError('Invalid GT geometry: ' + str(path))
            box = [max(0., (x-bw/2)*w), max(0., (y-bh/2)*h),
                   min(float(w), (x+bw/2)*w), min(float(h), (y+bh/2)*h)]
            area = (box[2]-box[0]) * (box[3]-box[1])
            anns.append(dict(gt_index=i, cls=int(c), bbox=[x,y,bw,bh], xyxy=box,
                             area_pixels=area, scale='small' if area < 32**2 else 'medium' if area < 96**2 else 'large'))
        people = [a for a in anns if a['cls'] == 14]
        others = [a for a in anns if a['cls'] != 14]
        pairs = []
        for a in people:
            aa = a['xyxy']
            for b in others:
                bb = b['xyxy']
                intersection = max(0,min(aa[2],bb[2])-max(aa[0],bb[0])) * max(0,min(aa[3],bb[3])-max(aa[1],bb[1]))
                gap = math.hypot(max(0,aa[0]-bb[2],bb[0]-aa[2]), max(0,aa[1]-bb[3],bb[1]-aa[3]))
                diag = math.hypot(aa[2]-aa[0],aa[3]-aa[1])
                pairs.append(dict(person_gt=a['gt_index'], nt_gt=b['gt_index'],
                                  intersection_over_person=intersection/a['area_pixels'],
                                  normalized_edge_gap=gap/max(diag,1e-12)))
        tags = []
        if not others: tags.append('person_only')
        for cls, name in [(11,'dog'),(12,'horse'),(1,'bicycle')]:
            if any(a['cls'] == cls for a in others): tags.append('person_'+name)
        if any(p['intersection_over_person'] >= .25 for p in pairs): tags.append('high_overlap')
        if pairs and all(p['intersection_over_person'] == 0 for p in pairs) and min(p['normalized_edge_gap'] for p in pairs) <= .1:
            tags.append('adjacent_no_overlap')
        tags += [scale+'_person' for scale in ('small','medium','large') if any(a['scale'] == scale for a in people)]
        if len(people) > 1: tags.append('multiple_person')
        records.append(dict(image_id=path.stem, image_path=str(image), label_path=str(path), width=w,height=h,
                            annotations=anns, pairs=pairs, strata=tags))
    required = ['person_only','person_dog','person_horse','person_bicycle','high_overlap','adjacent_no_overlap',
                'small_person','medium_person','large_person','multiple_person']
    selected = {}
    for tag in required:
        candidates = sorted((r for r in records if tag in r['strata']),
                            key=lambda r: hashlib.sha256(f'{seed}|{tag}|{r["image_id"]}'.encode()).hexdigest())
        if len(candidates) < 2:
            raise ValueError('Insufficient predefined stratum: '+tag)
        for row in candidates[:2]: selected[row['image_id']] = row
    remaining = sorted(records, key=lambda r: hashlib.sha256(f'{seed}|fill|{r["image_id"]}'.encode()).hexdigest())
    for row in remaining:
        if len(selected) >= count: break
        selected[row['image_id']] = row
    if len(selected) != count:
        raise ValueError('Requested sample size conflicts with strata coverage')
    result = sorted(selected.values(), key=lambda r:r['image_id'])
    for row in result:
        row['image_sha256'] = sha(row['image_path'])
        row['label_sha256'] = sha(row['label_path'])
    return dict(seed=seed, split='train', target_class=14, count=len(result),
                selection='first two per predefined stratum by SHA256(seed|stratum|id), then SHA256(seed|fill|id); no model/loss/AP used',
                geometry='original image pixel coordinates, continuous clipped GT boxes for selection; canonical rectangle for diagnostic masks',
                scale_definition='original GT clipped area small<32^2, medium<96^2, otherwise large; descriptive, not resized COCO evaluation',
                high_overlap_definition='any Person/NT intersection divided by that Person area >=0.25',
                adjacency_definition='all Person/NT boxes nonoverlapping and any edge gap/Person diagonal<=0.1',
                coverage={tag:sum(tag in r['strata'] for r in result) for tag in required},
                available_coverage={tag:sum(tag in r['strata'] for r in records) for tag in required}, records=result)


if __name__ == '__main__':
    import sys
    print(json.dumps(choose(sys.argv[1])))
