"""Output-only analysis; keeps static and independent-trajectory controls separate."""
import argparse,csv,json,statistics
from pathlib import Path


def average(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.input);out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    status=json.loads((root/'status.json').read_text());assert status['state']=='complete' and status['phase']=='measure' and status['new_optimizer_updates']==0
    rows=json.loads((root/'TRACE.json').read_text());pairs=json.loads((root/'STATIC_SPATIAL_PAIRS.json').read_text());assert len(rows)==60 and len(pairs)==320
    arms=['same_state_S0','Utility_replay','Static_Uniform_U'];oldconf=json.loads(Path('configs/tfss_short_probe.json').read_text())
    assert all(r['absolute_RMS_error']<=1e-7 and r['relative_RMS_error']<=1e-4 for r in pairs)
    summary={'comparison':'static same Utility pre-step state; original independent control remains failed at batch1 step5','source_commit':status['source_commit'],'new_optimizer_updates':0,'static_pairs':len(pairs),'max_abs_RMS_error':max(r['absolute_RMS_error'] for r in pairs),'max_rel_RMS_error':max(r['relative_RMS_error'] for r in pairs),'max_original_U_replay_difference':status['max_original_U_replay_difference'],'arms':{},'spatial':{},'target_GateB':{},'NT_GT':{},'candidate_decision':'REQUIRES_INTERPRETATION_NO_AUTOMATIC_FORMAL_RUN','limitations':['fixed TRAIN32, seed0, repeated steps/EOT not independent replicates','recorded pre-update states, no new optimization or victim/AP/convergence result','NT native clean/poison assignments may differ; signed loss changes are surrogate diagnostics','same_state_S0 is evaluated at Utility Fourier state, not the separately optimized historical S0']}
    curves=[]
    for r in rows:
        base=next(v for v in rows if (v['batch'],v['step'],v['eot'],v['arm'])==(r['batch'],r['step'],r['eot'],'same_state_S0'))
        protect=lambda v:v['weighted_L_BG']+v['weighted_L_rel']+v['L_adapt_target']
        curves.append({'batch':r['batch'],'step':r['step'],'eot':r['eot'],'arm':r['arm'],'L_total':r['L_total'],'L_BG':r['L_BG'],'L_rel':r['L_rel'],'L_ret':r['L_ret'],'protection':protect(r),'relative_protection_vs_same_state_S0':protect(r)/protect(base)-1,'target_clean_proxy':r['target_score_clean_after'],'target_poison_proxy':r['target_score_poison_after']})
    for arm in arms:
        rr=[r for r in rows if r['arm']==arm];pp=[p for r in rr for p in r['per_image']];cc=[r for r in curves if r['arm']==arm]
        m={'L_ret_mean':average(r['L_ret'] for r in rr),'protection_gradient_cosine_median':median(p['protection_gradient_cosine_to_s0'] for p in pp),'protection_gradient_norm_ratio_median':median(p['protection_gradient_norm_ratio_to_s0'] for p in pp),'high_protection_operator_energy_median':median(p['high_protection_operator_energy_retention'] for p in pp),'high_protection_undefined':sum(p['high_protection_operator_energy_retention'] is None for p in pp),'protection_relative_mean':average(r['relative_protection_vs_same_state_S0'] for r in cc),'protection_relative_by_batch':{str(b):average(r['relative_protection_vs_same_state_S0'] for r in cc if r['batch']==b) for b in [0,1]}}
        summary['arms'][arm]=m
        if arm!='same_state_S0':
            g=oldconf['gate_B_proxy_only'];checks={'mean_objective':m['protection_relative_mean']<=g['maximum_mean_relative_increase_BG_plus_rel_plus_adaptive_target'],'batch_objectives':max(m['protection_relative_by_batch'].values())<=g['maximum_batch_relative_increase'],'gradient_cosine':m['protection_gradient_cosine_median']>=g['minimum_median_protection_gradient_cosine'],'gradient_norm':m['protection_gradient_norm_ratio_median']>=g['minimum_median_gradient_norm_retention'],'high_protection_energy':m['high_protection_operator_energy_median']>=g['minimum_median_high_protection_operator_energy_retention']}
            summary['target_GateB'][arm]={'passed':all(checks.values()),'checks':checks,'anchor':'same Utility state S0'}
    flattened=[]
    for r in pairs:
        for region,s in r['regions'].items():flattened.append({k:r[k] for k in ['batch','step','eot','image_id']}|{'region':region}|s)
    # Average repeated forwards within each image first; report the number of
    # valid images instead of treating 320 repeated observations as independent.
    for region in pairs[0]['regions']:
        images=[]
        for b in [0,1]:
            for image_id in sorted({r['image_id'] for r in pairs if r['batch']==b}):
                rr=[r['regions'][region] for r in pairs if r['batch']==b and r['image_id']==image_id]
                u=average(r['Utility_energy_retention'] for r in rr);v=average(r['Uniform_energy_retention'] for r in rr)
                if u is not None and v is not None:images.append({'batch':b,'image_id':image_id,'Utility':u,'Uniform':v,'difference':u-v})
        summary['spatial'][region]={'valid_images':len(images),'Utility_mean_energy_retention':average(r['Utility'] for r in images),'Uniform_mean_energy_retention':average(r['Uniform'] for r in images),'paired_difference_mean':average(r['difference'] for r in images),'paired_difference_median':median(r['difference'] for r in images),'Utility_more_retention_images':sum(r['difference']>0 for r in images),'Utility_less_retention_images':sum(r['difference']<0 for r in images),'batch_mean_difference':{str(b):average(r['difference'] for r in images if r['batch']==b) for b in [0,1]},'per_image':images}
    gt=[]
    for b in [0,1]:
        for arm in arms:
            records=json.loads((root/f'batch{b}_step5/{arm}_NT_GT.json').read_text())
            for image in records:
                clean={(r['gt_index'],r['cls']):r for r in image['clean']};poison={(r['gt_index'],r['cls']):r for r in image['poison']};assert clean.keys()==poison.keys()
                for key,c in clean.items():
                    d=poison[key];assert c['positive_count']>0 and d['positive_count']>0
                    r={'batch':b,'image_id':image['image_id'],'GT_index':key[0],'class_id':key[1],'arm':arm}
                    for term in ['box','cls_bce19','dfl','weighted_loss']:r['clean_'+term]=c[term];r['poison_'+term]=d[term];r['delta_'+term]=d[term]-c[term]
                    gt.append(r)
    assert len(gt)==138
    classwise=[]
    for cls in sorted({r['class_id'] for r in gt}):
        r={'class_id':cls}
        for arm in arms:r[arm]=average(v['delta_weighted_loss'] for v in gt if v['class_id']==cls and v['arm']==arm)
        r['Utility_minus_static_Uniform']=r['Utility_replay']-r['Static_Uniform_U'];classwise.append(r)
    for arm in arms:
        rr=[r for r in gt if r['arm']==arm]
        summary['NT_GT'][arm]={'GT_observations':len(rr),'GT_mean_signed_loss_delta':average(r['delta_weighted_loss'] for r in rr),'equal_class_mean_signed_loss_delta':average(r[arm] for r in classwise),'GT_mean_by_batch':{str(b):average(r['delta_weighted_loss'] for r in rr if r['batch']==b) for b in [0,1]}}
    summary['NT_GT']['Utility_vs_static_Uniform']={'GT_mean_difference':summary['NT_GT']['Utility_replay']['GT_mean_signed_loss_delta']-summary['NT_GT']['Static_Uniform_U']['GT_mean_signed_loss_delta'],'lower_classes':sum(r['Utility_minus_static_Uniform']<0 for r in classwise),'higher_classes':sum(r['Utility_minus_static_Uniform']>0 for r in classwise),'classwise':classwise}
    table(out/'FORWARD_CURVES.csv',curves);table(out/'STATIC_SPATIAL_REGIONS.csv',flattened);table(out/'NT_GT_PAIRED.csv',gt);table(out/'NT_CLASSWISE.csv',classwise)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    import os
    os.environ['MPLCONFIGDIR']=str(out/'runtime/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for b,ax in enumerate(axes):
        for arm in arms:ax.plot(range(1,6),[average(r['L_ret'] for r in curves if r['batch']==b and r['step']==s and r['arm']==arm) for s in range(1,6)],marker='.',label=arm)
        ax.set(title=f'batch {b}: fixed Utility states',xlabel='saved pre-update state index',ylabel='NT feature drift / L_ret')
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'STATIC_L_RET.png',dpi=150);plt.close(fig)
    selected=['high_Q','high_R','high_R_low_Q','Q_weighted','R_weighted'];fig,ax=plt.subplots(figsize=(9,3.4));xx=list(range(len(selected)))
    ax.bar([x-.18 for x in xx],[summary['spatial'][r]['Utility_mean_energy_retention'] for r in selected],width=.36,label='Utility')
    ax.bar([x+.18 for x in xx],[summary['spatial'][r]['Uniform_mean_energy_retention'] for r in selected],width=.36,label='static matched uniform')
    ax.set_xticks(xx,selected);ax.set_ylabel('actual energy / SAME base energy');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'STATIC_SPATIAL_ENERGY.png',dpi=150);plt.close(fig)
    (out/'ANALYSIS_COMPLETE.json').write_text(json.dumps({'state':'complete','new_optimizer_updates':0,'model_forward':False,'source_RGB_exported':False}),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['spatial','NT_GT']},indent=2))


if __name__=='__main__':main()
