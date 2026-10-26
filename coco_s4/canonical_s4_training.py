"""Restore the canonical S4 optimization computation, with passive sink evidence.

This module starts no job and changes no canonical loss or optimizer source.
Actual learned-Teacher CPU/CUDA validation is required before formal admission.
"""
import hashlib
import json
import shutil
from pathlib import Path

CONTRACT = 'canonical-original-S4-TFSS-U-v1'
TFSS_LOCK = 'docs/S4_TFSS_UTILITY/S4_CONFIG_LOCK.json'
TFSS_LOCK_SHA256 = '33d3d7af6c76c2c881db0dfc7d6fc66dafa9b5655026c07332701143ed52d6ec'
MAP_VERSION = 'masked3x3-Q10Q90-clean-input-perGT-max-person-equalNTclass-image-canonical-canvas-v2'


def configure(trainer, manifest, output, root):
    import torch
    from ue_framework.methods.s4_tfss import S4Trainer, CleanMapCache
    from .runtime import sha, load
    if not isinstance(trainer, S4Trainer):
        raise ValueError('Original S4 requires its canonical S4Trainer computation')
    if sha(Path(root)/TFSS_LOCK) != TFSS_LOCK_SHA256 or manifest['files'].get(TFSS_LOCK) != TFSS_LOCK_SHA256:
        raise ValueError('Canonical Original S4 TFSS lock is absent or changed')
    lock = load(Path(root)/TFSS_LOCK)
    if lock['tfss']['eta_U'] != .85 or lock['tfss']['map_version'] != MAP_VERSION:
        raise ValueError('Original S4 TFSS formula identity changed')
    for name in ('s4_tfss.py', 'tfss_spatial.py', 'tfss_clean_maps.py', 'tfss_nt_sensitivity.py', 's4_optimization.py'):
        relative = 'ue_framework/methods/'+name
        if manifest['files'].get(relative) != sha(Path(root)/relative):
            raise ValueError('Canonical S4 computation is outside the bound source release')
    trainer.tfss_mode = 'TFSS-U'
    trainer.s4_cache = CleanMapCache(Path(output)/'clean_maps', dict(
        reference_sha256=manifest['teacher_dependency']['checkpoint_sha256'],
        maps_code_sha256=manifest['files']['ue_framework/methods/tfss_clean_maps.py'],
        risk_code_sha256=manifest['files']['ue_framework/methods/tfss_nt_sensitivity.py'],
        map_version=MAP_VERSION, canonical_lock_sha256=TFSS_LOCK_SHA256))
    trainer.tfss_draws = []
    trainer.tfss_total_draws = 0
    trainer.tfss_nonunit_draws = 0
    trainer.tfss_frozen_inputs = trainer.tfss_adaptive_inputs = 0
    def frozen_input(module, args):
        if args and args[0] is getattr(trainer, '_last_poison', None):
            expected = (trainer.tfss_input+trainer.tfss_delta).clamp(0, 1)
            if not torch.equal(args[0], expected):
                raise ValueError('Frozen loss did not consume the actual TFSS-weighted image')
            trainer.tfss_frozen_inputs += 1
    handle = trainer.surrogate.register_forward_pre_hook(frozen_input)
    original = trainer.adaptive.losses
    def adaptive_losses(clean, poison, items, masks, pooled):
        count = len(trainer.s4_item_paths)
        start = (trainer._adaptive_calls-1) % count
        indices = [(start+i) % count for i in range(min(trainer.adaptive_batch_size, count))]
        expected = (trainer.tfss_input+trainer.tfss_delta).clamp(0, 1)[indices]
        if not torch.equal(poison, expected):
            raise ValueError('Adaptive inner learner did not consume the actual TFSS-weighted image')
        trainer.tfss_adaptive_inputs += 1
        return original(clean, poison, items, masks, pooled)
    trainer.adaptive.losses = adaptive_losses
    def restore():
        handle.remove()
        trainer.adaptive.losses = original
    return restore


def audit_draw(trainer, before, *, retain):
    import torch
    weight = trainer.tfss_applied_weight
    if (weight.requires_grad or not torch.isfinite(weight).all()
            or bool(((weight < .85) | (weight > 1)).any())
            or not torch.equal(trainer.tfss_delta, trainer.tfss_s0_delta*weight)
            or trainer.tfss_frozen_inputs <= before[0] or trainer.tfss_adaptive_inputs != before[1]+1):
        raise ValueError('Canonical TFSS source-to-scaled-image/loss sink evidence failed')
    trainer.tfss_total_draws += 1
    trainer.tfss_nonunit_draws += int(bool((weight < 1).any()))
    if retain:
        trainer.tfss_draws.append(dict(image_ids=[Path(p).stem for p in trainer.s4_item_paths],
            W_min=float(weight.detach().min()), W_max=float(weight.detach().max()),
            weighted_delta_exact=True, frozen_scaled_input_observed=True,
            adaptive_scaled_input_observed=True,
            map_records_sha256=hashlib.sha256(json.dumps(trainer.s4_map_records,
                sort_keys=True,separators=(',', ':')).encode()).hexdigest()))


def proof(trainer, root):
    from .runtime import sha
    return dict(contract=CONTRACT, trainer='ue_framework.methods.s4_tfss.S4Trainer',
        optimizer='ue_framework.methods.s4_optimization.optimize_s4', TFSS_mode='TFSS-U', eta=.85,
        canonical_lock_sha256=TFSS_LOCK_SHA256, total_EOT_draws=trainer.tfss_total_draws,
        nonunit_weight_draws=trainer.tfss_nonunit_draws, frozen_scaled_inputs=trainer.tfss_frozen_inputs,
        adaptive_scaled_inputs=trainer.tfss_adaptive_inputs, retained_draws=trainer.tfss_draws,
        cache_identity=trainer.s4_cache.identity,
        canonical_sources={n:sha(Path(root)/n) for n in ('ue_framework/methods/s4_tfss.py',
            'ue_framework/methods/tfss_spatial.py','ue_framework/methods/s4_optimization.py')})


def validate_proof(value, *, draws):
    if (value.get('contract') != CONTRACT or value.get('TFSS_mode') != 'TFSS-U'
            or value.get('eta') != .85 or value.get('canonical_lock_sha256') != TFSS_LOCK_SHA256
            or value.get('optimizer') != 'ue_framework.methods.s4_optimization.optimize_s4'
            or value.get('trainer') != 'ue_framework.methods.s4_tfss.S4Trainer'
            or value.get('total_EOT_draws') != draws or value.get('nonunit_weight_draws', 0) <= 0
            or value.get('frozen_scaled_inputs', 0) < draws or value.get('adaptive_scaled_inputs') != draws
            or not value.get('retained_draws')):
        raise ValueError('Actual canonical Original S4 TFSS optimization proof is missing')
    for row in value['retained_draws']:
        if (not row.get('weighted_delta_exact') or not row.get('frozen_scaled_input_observed')
                or not row.get('adaptive_scaled_input_observed') or not .85 <= row['W_min'] <= row['W_max'] <= 1):
            raise ValueError('Invalid retained TFSS computation sink evidence')


def run(trainer, images, labels, prototypes, output, identity, *, bounded):
    """Use the original optimizer; package its artifacts without changing arithmetic."""
    import torch
    from ue_framework.methods.s4_optimization import optimize_s4
    from .protocol import expected_noise_schedule
    from .runtime import write_once
    native = Path(output)/'native_s4'
    schedule = expected_noise_schedule(trainer.num_classes, len(images))
    result = optimize_s4(trainer, images, labels, prototypes, native, identity,
        stop_after=1 if bounded else 0, formal_lock={'noise_schedule':schedule})
    def copy_once(source, destination):
        with Path(source).open('rb') as source_stream, Path(destination).open('xb') as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
    if bounded:
        # optimize_s4 deliberately returns at a completed-update boundary.
        # A bounded snapshot is not represented as a completed 24-epoch result.
        indices = trainer.freq_active_idx
        pack = dict(coords=[list(trainer.freq_candidate_coords[int(i)]) for i in indices.cpu().tolist()],
            fourier_coeff=trainer.fourier_coeff.detach()[indices].cpu(),
            suppress_small=trainer.suppress_small.detach().cpu(), method='legacy_kproto_ret',
            support_mode=trainer.support_mode, target_class_id=trainer.target_class_id, eps=trainer.eps,
            feature_layer=trainer.feature_layer, s4_identity=identity, num_prototypes=trainer.num_prototypes,
            margin=trainer.margin)
        with (Path(output)/'global_params.pt').open('xb') as stream:
            torch.save(pack, stream)
    else:
        copy_once(native/'global_params.pt', Path(output)/'global_params.pt')
    copy_once(native/'diagnostics.csv', Path(output)/'diagnostics.csv')
    write_once(Path(output)/'diagnostics.json',dict(latest=result['latest'],native_status=result,
        optimization_contract=CONTRACT, bounded=bool(bounded), official_result=not bounded))
    return result
