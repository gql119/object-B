"""Small final outputs derived from completed components, never model tuning."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time


def finalize(out,root,commit):
    out=Path(out);root=Path(root);reports=out/'reports';reports.mkdir()
    def read(p):return json.loads(Path(p).read_text())
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def dump(name,value):(reports/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    def md(name,text):(reports/name).write_text(text,encoding='utf8')
    lock=read(out/'config_lock_snapshot.json');audit=read(out/'source_audit.json')
    integrity=read(out/'dataset_integrity.json');noise=read(out/'noise/status.json')
    result=read(out/'comparison.json');completion=read(out/'victim_completion.json');metrics=result['metrics']
    if (noise.get('epochs_completed'),noise['steps'],completion['epochs'],integrity['state'])!=(24,9144,200,'passed'):
        raise ValueError('Report cannot claim incomplete components as complete')
    checkpoints=list((out/'noise/checkpoints').glob('*.pt'))
    epochs=[json.loads(x) for x in (out/'noise/epoch_summary.jsonl').read_text().splitlines()]
    if len(epochs)!=24:raise ValueError('Missing full epoch spatial/loss summaries')
    for source,name in [(out/'config_lock_snapshot.json','S4_CONFIG_LOCK.json'),(out/'dataset_integrity.json','S4_NOISE_INTEGRITY.json'),
        (out/'S4_evaluation.json','S4_LAST_EVALUATION.json'),(out/'S4_best_secondary_evaluation.json','S4_BEST_EVALUATION.json'),
        (out/'S4_TFSS_SPATIAL_STATS.csv','S4_TFSS_SPATIAL_STATS.csv'),(out/'voc20_comparison.csv','S4_VOC20_PER_CLASS.csv')]:shutil.copyfile(source,reports/name)
    md('S4_SOURCE_AUDIT.md','# Actual S4 source audit\n\n'+json.dumps(audit,indent=2)+'\n\nS4 Git SHA '+commit+'. Exact original45 S0 bytes verified at runtime. Historical S0 runtime Git SHA unknown. Parent/source config/reference/prototypes and channels are frozen. Only detached spatial Utility scaling changes the method.\n')
    md('S4_PREFLIGHT_REPORT.md','# S4 preflight\n\n'+json.dumps(dict(source_commit=commit,source=audit,config_sha256=sha(out/'config_lock_snapshot.json'),review=read(out/'review_snapshot.json')),indent=2)+'\n\nCurrent-source smoke/review admission is bound by the retained review receipt.\n')
    md('S4_NOISE_TRAINING_REPORT.md',f'# S4 full noise\n\n24 epochs / 9144 Adam updates completed, elapsed {noise["elapsed_seconds"]:.6f} seconds. Not a convergence proof. {len(checkpoints)} named periodic/epoch checkpoints retained, plus atomic latest-step continuation and append-only journal. All source/cursor/Fourier/Adam/scheduler/RNG identities are checked by the production optimizer.\n\nFinal checkpoint: {noise["checkpoint"]}\nSHA256: {sha(noise["checkpoint"])}\n\nProtection Fourier gradient is sampled on the first batch of each epoch (both EOT draws), not reported as measured on every step. Q/R/W distributions and region energies are batch/image descriptive summaries; fallback counts are repeated observations, not unique images. Full diagnostics.csv, spatial_per_image.jsonl, epoch_summary.jsonl remain in the output.\n')
    hist=list(csv.DictReader((out/'victim_run/victim/results.csv').open()))
    md('S4_VICTIM_TRAINING_REPORT.md','# S4 victim200\n\n'+json.dumps(dict(completion=completion,run_spec=read(out/'victim_run_spec.json'),contiguous_results_rows=len(hist)),indent=2)+'\n\nRandom YAML initialization; native model/loss/optimizer/augmentation path. Explicit patience0 honors full200 (no AP-based early stopping), original200epoch LR schedule unchanged. Finite guards prohibit native NaN repair/reduced-batch continuation. Native epoch checkpoints and periodic full precision state snapshots retained. Ultralytics final stripped epoch=-1 is serialization semantics, not a substitute for the verified200 CSV rows. No interrupted victim is silently resumed: exact worker/prefetch RNG continuity requires validation.\n')
    cols=['Method','Person_AP50','NT_mAP50','NT_mAP75','NT_mAP50_95','All_mAP50','All_mAP50_95','P','U','S','WCD','TCD3']
    rows=[]
    for name in ('Clean','S0','S1','S2','S3','S4'):
        m=metrics[name];rows.append(dict(zip(cols,[name,m['mAP50_target'],m['mAP50_non_target'],m['mAP75_non_target'],m['mAP50_95_non_target'],m['mAP50_all'],m['mAP50_95_all'],
            m['selectivity']['P'],m['selectivity']['U'],m['selectivity']['S'],m['collateral']['WCD'],m['collateral']['TCD3']])))
    with (reports/'S4_S0_S1_S2_S3_COMPARISON.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(rows)
    dump('S4_SELECTIVITY_METRICS.json',dict(primary='last.pt@200',secondary='best.pt',source_commit=commit,
        config_sha256=sha(out/'config_lock_snapshot.json'),outcome=result['outcome'],delta=result['delta'],
        metrics=metrics,thresholds=lock['success_gate'],PUS='original signed formulas relative to matched Clean; S harmonic mean',
        WCD_TCD3='original signed relative per-class AP50 loss; largest and mean top3, no clipping'))
    a=integrity['images']
    avg=lambda key:sum(v[key] for v in a)/len(a)
    s0_logs=read(root/'docs/S4_TFSS_UTILITY/S0_EPOCH_LOG_REFERENCE.json')
    if sha(s0_logs['path'])!=s0_logs['sha256'] or s0_logs['rows']!=9144:
        raise ValueError('Historical S0 loss reference changed')
    trajectories=[]
    for old,current in zip(s0_logs['epochs'],epochs):
        if old['epoch']!=current['epoch']:raise ValueError('S0/S4 epoch alignment mismatch')
        row=dict(epoch=old['epoch'])
        for k in ('L_BG','L_rel','L_ret','L_orth_objective','L_total','weighted_L_BG','weighted_L_rel','L_adapt_target'):
            row['S0_'+k]=old[k];row['S4_'+k]=current[k];row['S4_minus_S0_'+k]=current[k]-old[k]
        row['S0_protection_objective']=old['weighted_L_BG']+old['weighted_L_rel']+old['L_adapt_target']
        row['S4_protection_objective']=current['weighted_L_BG']+current['weighted_L_rel']+current['L_adapt_target']
        trajectories.append(row)
    with (reports/'S4_S0_LOSS_TRAJECTORY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trajectories[0]));writer.writeheader();writer.writerows(trajectories)
    mechanism=dict(decoded_dataset_RMS=dict(scope='mean per6095 Person images, actual decoded uint8 differences/255',
        S4_full_image_RMS=avg('rms_pixel'),S0_full_image_RMS=avg('S0_rms_pixel'),S4_Person_RMS=avg('rms_Person'),S0_Person_RMS=avg('S0_rms_Person')),
        spatial_epoch_evidence=epochs,observed_S0_loss_reference=dict(path=s0_logs['path'],sha256=s0_logs['sha256'],rows=9144,
            interpretation='descriptive observed trajectories, not paired causal effect; contemporaneous S0 runtime commit unknown'),
        loss_trajectory=trajectories,Person_Dog=result['person_to_dog'],AP50_changed_classes={
            name:metrics['S4']['AP50_per_class'][i]-metrics['S0']['AP50_per_class'][i]
            for i,name in enumerate(metrics['S4']['class_names'])},
        limits=['single seed, no statistical significance','full S4 vsS0 AP does not isolate attribution causally from overall attenuation',
                'Qt/Rnt are sensitivity proxies; no background-absorption claim','direct Person/Dog diagnostic is fixed-threshold GT overlap, not native one-to-one AP confusion'])
    dump('S4_MECHANISM_ANALYSIS.json',mechanism)
    md('S4_MECHANISM_ANALYSIS.md','# S4 mechanism evidence\n\n'+json.dumps(mechanism,indent=2)+'\n\nQuestions1/2: inspect all24 actual spatial summaries, not just early smoke. Question3: decoded RMS table above. Questions4/5: original S0 9144row CSV is SHA-bound, epoch means/protection objective and NT Lret differences are in S4_S0_LOSS_TRAJECTORY.csv; this is descriptive, not a matched-step causal control. Questions6/7/10/11/12: all20 class/target/NT/selectivity deltas are retained, including worsened classes. Question8: matched predeclared Person/Dog GT-overlap diagnostic, no categorical migration proof. Question9: W attenuates some overall energy; prior static same-RMS evidence supports local redistribution, but no additional full Uniform run exists. Never attribute all AP differences to the map.\n')
    table='|Method|Person AP50|NT AP50|NT AP75|All AP50|S|\n|---|---:|---:|---:|---:|---:|\n'
    for row in rows:table+='|'+row['Method']+'|'+ '|'.join('—' if row['Method']=='Clean' and k=='S' else f'{100*row[k]:.4f}' for k in ['Person_AP50','NT_mAP50','NT_mAP75','All_mAP50','S'])+'|\n'
    md('S4_FINAL_REPORT.md',f'# S4-TFSS-Utility formal seed0\n\nOutcome: {result["outcome"]}. Components complete; terminal controller completion and main-agent artifact acceptance are checked separately.\n\n{table}\n\nValues shown in percentage points; raw decimals and all20 AP50/AP75/AP50-95/Recall/Precision are saved in CSV/JSON. Main last.pt@200, best secondary. No parameter search/extra seed/CUP run.\n\nSource {commit}; config SHA {sha(out/"config_lock_snapshot.json")}; source dataset fingerprint {audit["dataset_fingerprint"]}; teacher SHA {lock["reference_checkpoint_sha256"]}; noise runtime {noise["elapsed_seconds"]:.6f}s; last/best SHA {completion["last_sha256"]}/{completion["best_sha256"]}.\n\nSource correctness, observed training logs, actual performance, mechanism hypotheses and unverified causal explanations are separate. This single seed does not establish significance or causality. See mechanism report for all limitations, uniform-control distinction, RMS and direct detection protocol. Existing data/weights/checkpoints retained remotely, never included in small publication.\n')
    files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in reports.iterdir() if p.is_file()}
    dump('S4_ARTIFACT_MANIFEST.json',dict(source_commit=commit,generated_at_unix=time.time(),files=files,self_excluded=True,
        raw_output_root=str(out),raw_component_files=['source_audit.json','config_lock_snapshot.json','review_snapshot.json','noise/status.json',
        'noise/diagnostics.csv','noise/spatial_per_image.jsonl','noise/epoch_summary.jsonl','noise/ordered_support_manifest.json','noise/checkpoints/',
        'noise/latest_continuation.pt','noise/diagnostics_journal.jsonl','noise/global_params.pt','clean_maps/','dataset/',
        'materialization_manifest.json','dataset_integrity.json','victim_run/victim/results.csv','victim_run/victim/weights/',
        'victim_run/victim/retained_epoch_checkpoints/','victim_run/victim/state_snapshots/','evaluation/','comparison.json'],
        checkpoint_cleanup='none; retain_all',publication='code/config/small reports only after terminal acceptance; no data/weights/secrets'))
