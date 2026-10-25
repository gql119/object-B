"""Finite CPU reconstruction of selected P5 coordinates from retained25K stats.

The component receipt accepts the reconstructed artifacts only. Formal use is
separately gated by current-source native CPU/CUDA evidence and one review.
No training image, validation image, or model forward is used here.
"""
import argparse
import copy
from pathlib import Path
import shutil

from .runtime import ROOT, DATA_ROOT, atomic, load, now, sha, verify_release, write_once
from .phase_contract import OWNED, ORIGIN_OUTPUT, teacher_dependency
from .phase_calibration import runtime_config, load_teacher
from .calibration import build_coordinates
from .dimension_protocol import PROTOCOL, RESULT_LABEL, validate_receipt_dimension

ORIGIN_RELEASE='5e16ca9f00af6e827cec3efea66b0ee930d90301ec95ac2d73a8b58ab04960ce'


def verify_reuse(binding):
    import torch
    source=Path(binding['source_root']).resolve();root=Path(binding['artifact_root']).resolve()
    if (not source.is_relative_to((OWNED/'releases').resolve()) or not root.is_relative_to(source)
            or sha(source/'source_release.json')!=ORIGIN_RELEASE):
        raise ValueError('Retained calibration release/root identity changed')
    paths={}
    for relative,item in binding['files'].items():
        p=Path(item['path']).resolve()
        if (p!=root/relative or not p.is_relative_to(root) or sha(p)!=item['sha256']
                or p.stat().st_size!=item['bytes']):
            raise ValueError('Retained calibration artifact hash/path changed: '+relative)
        paths[relative]=p
    for name in ('collected/raw_statistics.pt','collected/eligible_support_manifest.json','prototype_bank.pt',
                 'method/BLOCKED_PROTOCOL_COMPATIBILITY.json','method/reference/target_specific_channels.pt',
                 'method/reference/non_target_subspaces.pt','method/reference/coordinates_identity.json'):
        if name not in paths:raise ValueError('Missing retained input: '+name)
    raw=torch.load(paths['collected/raw_statistics.pt'],map_location='cpu',weights_only=True)
    support=load(paths['collected/eligible_support_manifest.json'])
    if (raw['images']!=25000 or raw['target_count']!=55096 or raw['validation_used']
            or support['paths']!=raw['eligible_person_images'] or len(support['paths'])!=13630
            or support['optimizer_steps']!=24*((len(support['paths'])+15)//16)):
        raise ValueError('Retained train-only count/support identity changed')
    classes=list(range(1,80))
    for layer in ('model.15','model.21'):
        item=raw['layers'][layer]
        if (sorted(item['pools'])!=classes or sorted(item['nt_counts'])!=classes
                or sorted(item['nt_sums'])!=classes or sum(item['nt_counts'].values())!=126379):
            raise ValueError('Retained calibration class completeness/count changed')
    block=load(paths['method/BLOCKED_PROTOCOL_COMPATIBILITY.json'])
    selected=block['actual_summary']['selected_channel_indices']
    if selected!=[r['index'] for r in binding['selected_channels']]:
        raise ValueError('Original selected-channel record changed')
    return raw,paths,selected


def run(release_path,output):
    import torch
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    from ue_framework.methods.semantic_coordinates import semantic_identity
    from ue_framework.methods.s4_tfss import model_hash
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise ValueError('Reconstruction is CPU-only')
    m=verify_release(release_path)
    if m.get('phase')!='dimension_rebuild' or m.get('dimension_protocol')!=PROTOCOL:
        raise ValueError('Wrong dimension reconstruction release')
    output=Path(output).resolve()
    if not output.is_relative_to((OWNED/'validation').resolve()) or output.exists():
        raise ValueError('Reconstruction requires a fresh isolated owned validation output')
    output.mkdir(parents=True,exist_ok=False)
    common=dict(phase='dimension_rebuild',source_commit=m['source_commit'],release_sha256=sha(release_path),
                dimension_protocol=PROTOCOL,result_label=RESULT_LABEL,official_metrics_disabled=True,
                artifact_ingest_disabled=True,validation_used=False,parameter_updates=False,cleanup_policy='retain_all',deleted_paths=[])
    atomic(output/'status.json',dict(common,state='RUNNING',UTC=now()))
    try:
        teacher_dependency(m['teacher_dependency'])
        evaluation=Path(m['teacher_eval_dependency']['accepted_path'])
        if sha(evaluation)!=m['teacher_eval_dependency']['accepted_sha256']:
            raise ValueError('Accepted Teacher evaluation receipt drift')
        er=load(evaluation)
        if (er['state']!='ACCEPTED' or er['image_count']!=5000 or er['classes']!=80
                or er['teacher_checkpoint_sha256']!=m['teacher_dependency']['checkpoint_sha256']):
            raise ValueError('Reconstruction requires the already accepted fixed Teacher evaluation')
        cfg,checkpoint,artifact_root=runtime_config(dict(artifact_relative_root=m['artifact_relative_root']),m)
        if cfg['methods']['legacy_kproto_ret']['semantic']['safe_bg'].get('dimension_protocol')!=PROTOCOL:
            raise ValueError('Production selected-space opt-in missing')
        raw,paths,selected=verify_reuse(m['retained_calibration'])
        artifact_root.mkdir(parents=True,exist_ok=False)
        (artifact_root/'collected').mkdir()
        shutil.copyfile(paths['collected/eligible_support_manifest.json'],artifact_root/'collected/eligible_support_manifest.json')
        shutil.copyfile(paths['prototype_bank.pt'],artifact_root/'prototype_bank.pt')
        reference={Path(name).name:dict(path=str(p),sha256=sha(p)) for name,p in paths.items() if name.startswith('method/reference/')}
        recovery_identity=dict(origin_release_sha256=ORIGIN_RELEASE,
                               raw_statistics_sha256=sha(paths['collected/raw_statistics.pt']),
                               original_selection_sha256=sha(paths['method/BLOCKED_PROTOCOL_COMPATIBILITY.json']),
                               prototype_bank_sha256=sha(paths['prototype_bank.pt']),
                               retained_input_binding_sha256=__import__('hashlib').sha256(
                                   __import__('json').dumps(m['retained_calibration'],sort_keys=True).encode()).hexdigest())
        coordinates=build_coordinates(raw,cfg,artifact_root/'method',builder_path=ROOT/'coco_s4/calibration.py',
                                     source_root=ROOT,reference_files=reference,
                                     expected_selection=selected,recovery_identity=recovery_identity)
        teacher,frozen=load_teacher(checkpoint,m['teacher_dependency']['checkpoint_sha256'],torch.device('cpu'))
        consumer=B2SemanticTrainer(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cpu'),
                                   copy.deepcopy(teacher),copy.deepcopy(teacher))
        dimension=len(coordinates['selected_channels'])
        if (consumer.channels.tolist()!=selected or len(consumer.safe_bases)!=79 or len(consumer.bases)!=79
                or consumer.orth_feature_layer!='model.15' or consumer.orth_channels.numel()!=48
                or consumer.safe_bg_rank!=8 or consumer.safe_bg_removal_alpha!=.5 or consumer.safe_bg_active_k!=3):
            raise ValueError('Actual selected-space/P3 consumer rejected reconstructed coordinates')
        bank=torch.load(artifact_root/'prototype_bank.pt',map_location='cpu',weights_only=True)
        if (bank['cache_key']!=consumer._cache_key(DATA_ROOT/'images/train',DATA_ROOT/'labels/train')
                or tuple(bank['prototypes'].shape)!=(8,raw['layers']['model.21']['target_sum'].numel())):
            raise ValueError('Retained K8 bank is not the actual selected-coordinate consumer bank')
        cache=consumer.safe_bg_cache
        clean=bank['prototypes'][:1].index_select(1,consumer.channels)
        background=bank['prototypes'][1:2].index_select(1,consumer.channels)
        from ue_framework.methods.safebg import safe_background_direction
        projected,diagnostic=safe_background_direction(clean,background,consumer.safe_bases,active_k=3,
            mode='low_rank',safe_rank=8,removal_alpha=.5,triplet_cache=cache)
        if (not torch.isfinite(projected).all() or int(diagnostic['projector_rank'][0])!=8
                or diagnostic['projection_ratios'].shape!=(1,79)):
            raise ValueError('Actual reconstructed79-class SafeBG projector is invalid')
        if model_hash(teacher)!=frozen or sha(checkpoint)!=m['teacher_dependency']['checkpoint_sha256']:
            raise ValueError('Read-only reconstruction changed the accepted Teacher')
        semantic=semantic_identity(cfg,ROOT)
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
               for p in sorted(artifact_root.rglob('*')) if p.is_file()]
        receipt=dict(common,phase='calibration',state='ACCEPTED',acceptance_scope='CPU_RECONSTRUCTION_COMPONENT',
                     formal_use_requires_current_review=True,scientific_review_complete=False,
                     teacher_checkpoint_sha256=m['teacher_dependency']['checkpoint_sha256'],
                     teacher_receipt_sha256=m['teacher_dependency']['accepted_sha256'],
                     teacher_eval_receipt_sha256=m['teacher_eval_dependency']['accepted_sha256'],
                     checkpoint_selection_allowed=False,images=25000,person_ground_truth=55096,
                     selected_channels=selected,selected_channel_count=dimension,
                     total_feature_channels=raw['layers']['model.21']['target_sum'].numel(),
                     eligible_channel_count=load(artifact_root/'method/coordinates/strict_tsfl_summary.json')['safe_candidates'],
                     non_target_objects_by_class={str(c):raw['layers']['model.21']['nt_counts'][c] for c in range(1,80)},
                     accepted_clean_train_manifest_sha256=raw['clean_train_manifest_sha256'],
                     eligible_person_image_count=len(raw['eligible_person_images']),
                     original_fourier_optimizer_steps=24*((len(raw['eligible_person_images'])+15)//16),
                     coordinates=coordinates,semantic_identity=semantic,artifact_files=files,
                     prototypes=dict(path=str(artifact_root/'prototype_bank.pt'),sha256=sha(artifact_root/'prototype_bank.pt'),
                                     cache_key=bank['cache_key'],shape=list(bank['prototypes'].shape),count=bank['count']),
                     reused_identity=recovery_identity,P3_reference_byte_reused=True,model_forwards=0,GPU_initialized=False,
                     cache_memory=cache.memory_report(),completed_UTC=now())
        validate_receipt_dimension(receipt)
        write_once(output/'candidate_receipt.json',receipt)
        write_once(output/'phase_spec.json',dict(source_root=str(ROOT),artifact_root=str(artifact_root),
                   release_sha256=sha(release_path),dimension_protocol=PROTOCOL))
        atomic(output/'status.json',dict(common,state='passed',selected_count=dimension,
               receipt_sha256=sha(output/'candidate_receipt.json'),GPU_initialized=False,model_forwards=0,completed_UTC=now()))
        return receipt
    except BaseException as exc:
        atomic(output/'status.json',dict(common,state='FAILED',error=repr(exc),ended_UTC=now()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.release,a.output)
