"""Read-only independent Full acceptance adapter; never changes the Full run."""
import csv,json,math
from pathlib import Path
from .common import load,sha
from .metrics import paired

def full_state(config):
    p=Path(config['predecessor']['output'])/'pipeline_status.json'
    return load(p) if p.is_file() else None

def verify_predecessor(config):
    """Called only AFTER native complete. Hashes actual checkpoints/files, not a Finished log."""
    pre=config['predecessor'];out=Path(pre['output']);state=full_state(config)
    if not state or state.get('state')!='complete':return None
    if (state.get('run_id'),state.get('source_commit'),state.get('config_sha256'))!=(pre['run_id'],pre['source_commit'],pre['config_sha256']):raise ValueError('Predecessor identity mismatch')
    if sha(out/'config_lock_snapshot.json')!=pre['config_sha256']:raise ValueError('Predecessor lock mutated')
    lock=load(out/'config_lock_snapshot.json');noise=load(out/'noise/status.json');done=load(out/'victim_completion.json')
    if not (noise.get('schedule_complete') is True and noise['steps']==9144 and done['epochs']==done['rows']==200):raise ValueError('Incomplete Full schedules')
    integrity=load(out/'dataset_integrity.json')
    if sha(out/'noise/global_params.pt')!=integrity['noise_checkpoint_sha256']:raise ValueError('Full generated noise identity changed')
    for key in ['labels_byte_identical','clean_val_byte_identical','outside_target_zero','per_pixel_frozen_budget_passed','decoded_PNG_audited']:
        if integrity.get(key) is not True:raise ValueError('Full data audit missing '+key)
    if (integrity['train'],integrity['val'],integrity['person_gt'],integrity['changed_person_gt'])!=(16551,4952,13256,13256):raise ValueError('Full dataset incomplete')
    manifest=out/'materialization_manifest.json'
    if sha(manifest)!=integrity['materialization_manifest_sha256']:raise ValueError('Full manifest mismatch')
    rows=load(manifest)
    if len(rows)!=16551 or len({r['id'] for r in rows})!=16551:raise ValueError('Full manifest IDs')
    for r in rows:
        p=Path(r['output'])
        if not p.is_relative_to(out/'dataset/images/train') or sha(p)!=r['output_sha256']:raise ValueError('Full training data changed')
        if sha(out/'dataset/labels/train'/(r['id']+'.txt'))!=r['label_sha256']:raise ValueError('Full labels changed')
    clean=Path(config['baseline']['clean_root'])
    for p in (out/'dataset/images/val').iterdir():
        if p.is_file() and sha(p)!=sha(clean/'images/val'/p.name):raise ValueError('Full clean val changed')
    for p in (out/'dataset/labels/val').glob('*.txt'):
        if sha(p)!=sha(clean/'labels/val'/p.name):raise ValueError('Full val label changed')
    if len(list((out/'dataset/images/val').iterdir()))!=4952:raise ValueError('Full val count')
    csvpath=out/'victim_run/victim/results.csv';history=list(csv.DictReader(csvpath.open()))
    if len(history)!=200 or [int(float(next(iter(r.values())))) for r in history]!=list(range(1,201)):raise ValueError('Missing Full epoch200')
    if any(not math.isfinite(float(v)) for r in history for k,v in r.items() if 'loss' in k):raise ValueError('Nonfinite Full loss')
    proof={}
    import torch
    torch.set_num_threads(1)
    for name,role,key in [('last.pt','S4_TFSS_FULL_OVERLAP','last_sha256'),('best.pt','S4_TFSS_FULL_OVERLAP_best_secondary','best_sha256')]:
        path=out/'victim_run/victim/weights'/name;digest=sha(path);metrics=load(out/(role+'_evaluation.json'))
        if digest!=done[key] or metrics['checkpoint_sha256']!=digest:raise ValueError('Full checkpoint attribution')
        paired(metrics,metrics) # validates every AP/Recall/Precision value
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(v).all() for v in ck['model'].state_dict().values() if torch.is_tensor(v)):raise ValueError('Nonfinite Full model')
        if name=='last.pt':
            embedded=ck['train_results'];epochs=next(v for k,v in embedded.items() if k.strip()=='epoch')
            if len(epochs)!=200 or int(float(epochs[-1]))!=200:raise ValueError('Full embedded epoch identity')
        del ck
        proof[name]=digest
    statefile=out/'victim_run/victim/state_snapshots/epoch_200_full_state.pt'
    if not statefile.is_file():raise ValueError('Full optimizer/RNG snapshot missing')
    snapshot=torch.load(statefile,map_location='cpu',weights_only=False)
    if snapshot['epoch']!=200 or not all(k in snapshot for k in ('optimizer','scheduler','scaler','rng')):raise ValueError('Full final state incomplete')
    del snapshot
    compare=load(out/'comparison.json')
    if compare['checkpoint_sha256']!=done['last_sha256'] or compare['source_commit']!=pre['source_commit']:raise ValueError('Full report attribution')
    if not (out/'S4_VS_TFSS_FULL_REPORT.md').is_file():raise ValueError('Full report missing')
    return dict(state='COMPLETED_AND_ACCEPTED',adapter='independent read-only evidence adapter; native Full state remains unchanged',run_id=pre['run_id'],source_commit=pre['source_commit'],config_sha256=pre['config_sha256'],checkpoints=proof,noise_steps=9144,victim_epochs=200,dataset_manifest_sha256=sha(manifest),validation_count=4952,comparison_sha256=sha(out/'comparison.json'),native_state_sha256=sha(out/'pipeline_status.json'),full_state_sha256=sha(statefile))
