"""Derived descriptive analysis of a completed bounded comparison; no model."""
import csv,json,math,statistics,sys
from pathlib import Path

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
status=json.loads((root/'status.json').read_text());assert status['state']=='partial' and status['phase']=='short' and len(status['trials'])==12
cfg=json.loads(Path('configs/tfss_short_probe.json').read_text());gate=cfg['gate_B_proxy_only']
mean=statistics.mean
def med(xs):
    xs=[x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None
def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
traces={};equiv=[];maps=[];summary={'scope':'two fixed independent TRAIN batches, eachmethod freshseed0/5updates/EOT2; descriptive only, no victim/AP','source_commit':status['source_commit'],'source_commits_by_batch':status.get('source_commits_by_batch',{}),'run_state':'PARTIAL_59_OF_60_UPDATES','candidate_decision':'INSUFFICIENT_EVIDENCE','methods':{},'GateA':{'passed':False,'completeness':'FAIL: batch1 Uniform-U step5 unreachable','equivalence_maxdiff':0.,'RMS_max_absolute':0.,'RMS_max_relative':0.},'GateB':{},'GateC':{},'map_costs':{}}
methods=['S0-reference','S0-noop','TFSS-C','TFSS-U','Uniform-C','Uniform-U']
for b in [0,1]:
    e=json.loads((root/f'batch{b}_S0-noop/equivalence.json').read_text());assert e['max_difference']<=1e-7;equiv.append(e['max_difference'])
    maps.extend(json.loads((root/f'maps_batch{b}.json').read_text()))
    for method in methods:
        rows=json.loads((root/f'batch{b}_{method}/trace.json').read_text());n=4 if b==1 and method=='Uniform-U' else 5;assert len(rows)==n*2 and [(x['step'],x['eot']) for x in rows]==[(s,e) for s in range(1,n+1) for e in [1,2]]
        assert all(math.isfinite(r['L_total']) and r['outside_support_max']==0 for r in rows)
        traces[b,method]=rows
summary['GateA']['equivalence_maxdiff']=max(equiv)
curves=[];per_image=[];rms=[];gtrows=[]
for (b,method),rows in traces.items():
    base=traces[b,'S0-reference']
    for row,ref in zip(rows,base):
        protection=row['weighted_L_BG']+row['weighted_L_rel']+row['L_adapt_target']
        refp=ref['weighted_L_BG']+ref['weighted_L_rel']+ref['L_adapt_target']
        curves.append({'batch':b,'method':method,'step':row['step'],'eot':row['eot'],'L_total':row['L_total'],'L_BG':row['L_BG'],'L_rel':row['L_rel'],'L_ret':row['L_ret'],'protection_objective':protection,'protection_relative_vs_S0':protection/refp-1,'non_target_feature_drift':row['non_target_feature_drift'],'target_clean_proxy':row['target_score_clean_after'],'target_poison_proxy':row['target_score_poison_after']})
        for p,q in zip(row['per_image'],ref['per_image']):
            v={'batch':b,'method':method,'step':row['step'],'eot':row['eot'],'image_id':row['fixed_image_ids'][p['image_index']]}
            for k in ['protection_gradient_cosine_to_s0','protection_gradient_norm_ratio_to_s0','high_protection_operator_energy_retention','fourier_protection_cosine_to_s0','weight_mean','weight_smoothness','effective_area']:v[k]=p[k]
            for k in ['person','overlap','exclusive','high_tsfl','high_nt_risk']:
                v[k+'_RMS']=p[k]['rms'];v[k+'_mean_abs']=p[k]['mean_abs']
            v['paired_person_RMS_reduction_fraction']=1-p['person']['rms']/q['person']['rms'];per_image.append(v)
        if method.startswith('Uniform-'):
            pair=traces[b,'TFSS-'+method[-1]][(row['step']-1)*2+row['eot']-1]
            for i,(x,y) in enumerate(zip(row['actual_person_rms'],pair['actual_person_rms'])):
                err=abs(x-y);rel=err/y;assert err<=cfg['uniform_tolerance_absolute_rms'] and rel<=cfg['uniform_tolerance_relative_rms']
                rms.append({'batch':b,'method':method,'step':row['step'],'eot':row['eot'],'image_id':row['fixed_image_ids'][i],'candidate_RMS':y,'uniform_RMS':x,'absolute_error':err,'relative_error':rel})
    ntpath=root/f'batch{b}_{method}/nt_gt_diagnostics.json';records=json.loads(ntpath.read_text()) if ntpath.exists() else [];assert records or (b==1 and method=='Uniform-U')
    for im in records:
        clean={(p['gt_index'],p['cls']):p for p in im['clean']}
        assert set(clean)=={(p['gt_index'],p['cls']) for p in im['poison']}
        for p in im['poison']:
            q=clean[p['gt_index'],p['cls']];assert p['positive_count']>0 and q['positive_count']>0
            r={'batch':b,'method':method,'image_id':im['image_id'],'gt_index':p['gt_index'],'class_id':p['cls'],'forward':im['forward']}
            for term in ['box','cls_bce19','dfl','weighted_loss']:
                r['clean_'+term]=q[term];r['poison_'+term]=p[term];r['delta_'+term]=p[term]-q[term]
            gtrows.append(r)
summary['GateA'].update(RMS_pair_cases=len(rms),RMS_max_absolute=max(x['absolute_error'] for x in rms),RMS_max_relative=max(x['relative_error'] for x in rms))
for method in methods:
    rs=[x for x in curves if x['method']==method];ps=[x for x in per_image if x['method']==method];ng=[x for x in gtrows if x['method']==method]
    summary['methods'][method]={'mean_person_RMS':mean(x['person_RMS'] for x in ps),'mean_paired_person_RMS_reduction':mean(x['paired_person_RMS_reduction_fraction'] for x in ps),'median_gradient_cosine':med([x['protection_gradient_cosine_to_s0'] for x in ps]),'median_gradient_norm_retention':med([x['protection_gradient_norm_ratio_to_s0'] for x in ps]),'median_high_protection_operator_energy_retention':med([x['high_protection_operator_energy_retention'] for x in ps]),'high_protection_undefined_observations':sum(x['high_protection_operator_energy_retention'] is None for x in ps),'mean_protection_relative_increase':mean(x['protection_relative_vs_S0'] for x in rs),'batch_protection_relative_increase':{str(b):mean(x['protection_relative_vs_S0'] for x in rs if x['batch']==b) for b in [0,1]},'mean_L_ret':mean(x['L_ret'] for x in rs),'mean_non_target_feature_drift':mean(x['non_target_feature_drift'] for x in rs),'GT_weighted_loss_delta_mean':mean(x['delta_weighted_loss'] for x in ng),'final_NT_batches':sorted({x['batch'] for x in ng}), 'recorded_EOT_rows':len(rs),'NT_GT_observations':len(ng)}
for method in ['TFSS-C','TFSS-U']:
    m=summary['methods'][method];checks={'mean_protection_objective':m['mean_protection_relative_increase']<=gate['maximum_mean_relative_increase_BG_plus_rel_plus_adaptive_target'],'each_batch_objective':max(m['batch_protection_relative_increase'].values())<=gate['maximum_batch_relative_increase'],'gradient_cosine':m['median_gradient_cosine']>=gate['minimum_median_protection_gradient_cosine'],'gradient_norm':m['median_gradient_norm_retention']>=gate['minimum_median_gradient_norm_retention'],'high_protection_energy':m['median_high_protection_operator_energy_retention'] is not None and m['median_high_protection_operator_energy_retention']>=gate['minimum_median_high_protection_operator_energy_retention']}
    summary['GateB'][method]={'passed':all(checks.values()),'checks':checks}
    u=summary['methods']['Uniform-'+method[-1]];s0=summary['methods']['S0-reference']
    summary['GateC'][method]={'completeness':'COMPLETE' if method=='TFSS-C' else 'INCOMPLETE: batch1 Uniform-U final NTGT/step5 missing','mean_Lret_vs_S0':m['mean_L_ret']-s0['mean_L_ret'],'mean_Lret_vs_Uniform':m['mean_L_ret']-u['mean_L_ret'] if method=='TFSS-C' else None,'GT_loss_delta_vs_S0':m['GT_weighted_loss_delta_mean']-s0['GT_weighted_loss_delta_mean'],'GT_loss_delta_vs_Uniform':m['GT_weighted_loss_delta_mean']-u['GT_weighted_loss_delta_mean'] if method=='TFSS-C' else None,'interpretation':'negative is lower recorded proxy; native clean/poison GT assignments may differ; no statistical/causal/AP claim'}
summary['map_costs']={'Qt_seconds_total':sum(x['q_target_seconds'] for x in maps),'extra_NT_risk_seconds_total':sum(x['extra_nt_risk_seconds'] for x in maps),'no_NT_images':sum(x['no_nt_gt'] for x in maps),'Person_GT':sum(len(x['target_objects']) for x in maps),'NT_GT':sum(len(x['nt_matching']) for x in maps),'NT_classes_observed':sorted({r['cls'] for x in maps for r in x['nt_matching']}),'Qt_invalid_GT':sum(not r['valid'] for x in maps for r in x['target_objects'])}
classwise=[]
for cls in sorted({x['class_id'] for x in gtrows}):
    for method in methods:
        rs=[x for x in gtrows if x['class_id']==cls and x['method']==method]
        classwise.append({'class_id':cls,'method':method,'GT_observations':len(rs),'GT_weighted_loss_delta_mean':mean(x['delta_weighted_loss'] for x in rs) if rs else None})

for collection in [curves,per_image,rms,gtrows]:
    for row in collection:row['source_commit']=status.get('source_commits_by_batch',{}).get(str(row['batch']),status['source_commit'])
summary['map_costs']['NT_risk_degenerate_class_maps']=sum(not r['valid'] for x in maps for r in x['risk_classes'])
csvout('PER_STEP_CURVES.csv',curves);csvout('PER_IMAGE_GRADIENT_STRENGTH.csv',per_image);csvout('ACTUAL_RMS_PAIRING.csv',rms);csvout('NON_TARGET_GT.csv',gtrows);csvout('NON_TARGET_CLASSWISE.csv',classwise)
(out/'GATE_A_B_C_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({'run_state':summary['run_state'],'methods':summary['methods'],'GateA':summary['GateA'],'GateB':summary['GateB'],'GateC':summary['GateC'],'map_costs':summary['map_costs']},indent=2))
