import json,csv,statistics
from pathlib import Path
p=Path('docs/tfss/short_comparison_20261017');root=p/'raw';out=p/'analysis'
def mean(xs):
    values=[x for x in xs if x is not None]
    return statistics.mean(values) if values else None
def med(xs):
    values=[x for x in xs if x is not None]
    return statistics.median(values) if values else None
methods=['S0-reference','S0-noop','TFSS-C','TFSS-U','Uniform-C','Uniform-U']
traces={(b,m):json.loads((root/f'batch{b}_{m}/trace.json').read_text()) for b in [0,1] for m in methods}
summary={'limitations':'32 fixed images; repeated EOT and steps are not independent replicates; undefined cosine/regions excluded and raw records retained; U matched windows only, not a complete comparison','methods':{},'classwise':[],'Uniform_U_matched_windows':{}}
for m in ['TFSS-C','TFSS-U','Uniform-C']:
 rows=[]
 for b in [0,1]:
  for i in range(16):
   xs=[r['per_image'][i] for r in traces[b,m]];base=[r['per_image'][i] for r in traces[b,'S0-reference']]
   row={'batch':b,'image_id':traces[b,m][0]['fixed_image_ids'][i],'RMS_reduction':mean(1-x['person']['rms']/y['person']['rms'] for x,y in zip(xs,base)),'gradient_cosine_mean':mean(x['protection_gradient_cosine_to_s0'] for x in xs)}
   for region in ['overlap','exclusive','high_tsfl','high_nt_risk']:
    pairs=[(x[region]['rms'],y[region]['rms']) for x,y in zip(xs,base) if x[region]['rms'] is not None and y[region]['rms']]
    row[region+'_RMS_reduction']=mean(1-x/y for x,y in pairs)
   rows.append(row)
 values=[r['RMS_reduction'] for r in rows]
 summary['methods'][m]={'RMS_reduction_across32_images':{'min':min(values),'median':med(values),'max':max(values),'positive_images':sum(x>0 for x in values)},'batch_grad_cosine_medians':{str(b):med(r['protection_gradient_cosine_to_s0'] for t in traces[b,m] for r in t['per_image']) for b in [0,1]},'region_mean_per_image_RMS_reduction':{k:mean(r[k+'_RMS_reduction'] for r in rows) for k in ['overlap','exclusive','high_tsfl','high_nt_risk']},'per_image':rows}
for b in [0,1]:
 control=traces[b,'Uniform-U'];candidate=traces[b,'TFSS-U'][:len(control)]
 summary['Uniform_U_matched_windows'][str(b)]={'steps':len(control)//2,'Lret_candidate':mean(x['L_ret'] for x in candidate),'Lret_control':mean(x['L_ret'] for x in control),'difference':mean(x['L_ret']-y['L_ret'] for x,y in zip(candidate,control))}
gt=list(csv.DictReader((out/'NON_TARGET_GT.csv').open(encoding='utf-8-sig')))
names=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
for cls in sorted({int(x['class_id']) for x in gt}):
 row={'class_id':cls,'class_name':names[cls]}
 for m in ['S0-reference','TFSS-C','TFSS-U','Uniform-C']:
  row[m]=mean(float(x['delta_weighted_loss']) for x in gt if int(x['class_id'])==cls and x['method']==m)
 row.update(C_minus_S0=row['TFSS-C']-row['S0-reference'],C_minus_UniformC=row['TFSS-C']-row['Uniform-C'],U_minus_S0=row['TFSS-U']-row['S0-reference'])
 summary['classwise'].append(row)
summary['class_sign_counts']={key:{'lower':sum(r[key]<0 for r in summary['classwise']),'higher':sum(r[key]>0 for r in summary['classwise']),'equal':sum(r[key]==0 for r in summary['classwise'])} for key in ['C_minus_S0','C_minus_UniformC','U_minus_S0']}
(out/'ADDITIONAL_PAIRED_DIAGNOSTICS.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='methods'},indent=2))
print(json.dumps({k:{i:j for i,j in v.items() if i!='per_image'} for k,v in summary['methods'].items()},indent=2))
