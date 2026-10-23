"""One fixed train-only calibration, following CUP_S4_CALIBRATION_RULE.md."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def quantile(values, q):
    values=sorted(values)
    x=(len(values)-1)*q
    i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def calibrate(source, output):
    source,output=Path(source),Path(output)
    if output.exists():raise ValueError('Preserve existing calibration')
    rows=json.loads((source/'retention_classes.json').read_text())
    grouped=defaultdict(list)
    for row in rows:
        if row['role']=='P3_base' and row['layer']=='P3' and not row['post_five_updates']:
            grouped[(row['batch'],row['step'],row['cls'])].append(row['raw_loss'])
    if any(len(v)!=2 for v in grouped.values()):raise ValueError('Require paired EOT2 class observations')
    byclass=defaultdict(list)
    for (_,_,cls),values in grouped.items():byclass[cls].append(mean(values))
    scales={k:max(median(v),1e-12) for k,v in byclass.items()}
    normalized=[value/scales[k] for k,values in byclass.items() for value in values]
    root=Path(__file__).resolve().parents[1]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(schema='s5.cup-calibration.v1',layer='P3',beta=.95,
        temperature=max(quantile(normalized,.75)-quantile(normalized,.25),1.),
        damage_clip=max(quantile(normalized,.99),1.),min_weight_factor=.5,max_weight_factor=2.,
        normalization_floor=1e-12,known_scales=scales,
        scale_observation_counts={k:len(v) for k,v in byclass.items()},
        unobserved_NT_classes=[k for k in range(20) if k!=14 and k not in scales],
        unobserved_class_policy='first complete outer-step EOT-mean anchors fixed scale; neutral priority1 until observed',
        source_run_id=source.name,source_commit=json.loads((source/'layer_probe_summary.json').read_text())['source_commit'],
        source_artifact_sha256={p:sha(source/p) for p in ('retention_classes.json','layer_probe_summary.json','artifact_manifest.json')},
        rule_sha256=sha(root/'docs/S5_CUP_S4/CUP_S4_CALIBRATION_RULE.md'),
        calibration_updates=0,model_forwards=0,validation_or_victim_AP_used=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();calibrate(a.source,a.output)
