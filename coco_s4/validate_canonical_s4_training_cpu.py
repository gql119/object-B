"""Component checks only: canonical optimizer adapter, no learned Teacher/data/GPU."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import canonical_s4_training as wiring


class ToyTrainer:
    """A declared scalar-loss fixture for the actual canonical optimizer loop."""
    num_classes = 80
    universal_epochs = 24
    universal_batch_size = 16
    eot_samples = 2
    universal_lr_fourier = .001
    enable_adaptive_freq_basis = True
    adaptive_freq_warmup_epochs = 5
    tfss_mode = 'S0'
    worst_k = 3
    _adaptive_calls = _semantic_calls = 0
    eps = 8/255
    target_class_id = 0
    support_mode = 'bbox_half_overlap'
    feature_layer = 'model.21'
    num_prototypes = 8
    margin = .2

    def __init__(self):
        self.surrogate = torch.nn.Linear(1, 1)
        self.adaptive = SimpleNamespace(model=copy.deepcopy(self.surrogate))
        self.fourier_coeff = torch.nn.Parameter(torch.full((64, 3), .01))
        self.suppress_small = torch.zeros(1, 3, 32, 32)
        self.freq_score = torch.zeros(64)
        self.freq_usage = torch.zeros(64)
        self.freq_active_idx = torch.arange(16)
        self.freq_candidate_coords = [(i, 1) for i in range(64)]
        self.class_names = ['class_'+str(i) for i in range(80)]
        self.bases = dict.fromkeys(range(1, 80))
        self.observed_paths = []

    def _get_active_freq_indices(self, epoch, step):
        return self.freq_active_idx

    def _build_support(self, shape, annotations, **kwargs):
        support = np.ones(shape[:2], np.float32)
        return support, support, support

    def _batch_loss(self, items, prototypes):
        self.observed_paths.append(list(self.s4_item_paths))
        loss = self.fourier_coeff.square().mean()
        metrics = dict(L_total=loss, selected_proto_ids='0')
        metrics.update({'L_orth_'+self.class_names[i]:0. for i in range(1, 80)})
        return loss, metrics


def main(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if not output.is_relative_to((root/'execution').resolve()) or output.exists():
        raise ValueError('Use a new retained execution output directory')
    output.mkdir(parents=True, exist_ok=False)
    changed = ['canonical_s4_training.py', 'phase_fourier.py', 'fourier_gradient_audit.py',
               'dimension_release.py', 'fourier_release.py', 'phase_export.py']
    for name in changed:
        p = root/'coco_s4'/name
        compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    tree = ast.parse((root/'coco_s4/phase_fourier.py').read_text(encoding='utf-8'))
    classes = {n.name:[ast.unparse(b) for b in n.bases] for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert classes['EligibleCOCOS4Trainer'] == ['S4Trainer']
    assert 'EligibleCOCOB2Trainer' not in classes
    rejected = False
    try:
        wiring.validate_proof({}, draws=2)
    except ValueError:
        rejected = True
    assert rejected, 'Old successful B2 smoke must fail the TFSS admission gate'
    fixture_proof = dict(contract=wiring.CONTRACT, TFSS_mode='TFSS-U', eta=.85,
        trainer='ue_framework.methods.s4_tfss.S4Trainer',
        optimizer='ue_framework.methods.s4_optimization.optimize_s4',
        canonical_lock_sha256=wiring.TFSS_LOCK_SHA256, total_EOT_draws=2,
        nonunit_weight_draws=2, frozen_scaled_inputs=2, adaptive_scaled_inputs=2,
        retained_draws=[dict(W_min=.85,W_max=1.,weighted_delta_exact=True,
            frozen_scaled_input_observed=True,adaptive_scaled_input_observed=True)])
    wiring.validate_proof(fixture_proof, draws=2)
    rejected_proofs = 0
    for key, value in (('total_EOT_draws',1), ('nonunit_weight_draws',0),
                       ('adaptive_scaled_inputs',1), ('eta',.8), ('trainer','B2SemanticTrainer')):
        invalid = copy.deepcopy(fixture_proof); invalid[key] = value
        try:
            wiring.validate_proof(invalid, draws=2)
        except ValueError:
            rejected_proofs += 1
    assert rejected_proofs == 5
    from ue_framework.methods.b2_semantic import B2SemanticTrainer
    rejected = False
    try:
        wiring.configure(object.__new__(B2SemanticTrainer), {}, output, root)
    except ValueError:
        rejected = True
    assert rejected
    trainer = ToyTrainer()
    before = trainer.fourier_coeff.detach().clone()
    images = [str(output/('image_'+str(i)+'.jpg')) for i in range(20)]
    loaded = []
    def fixture_image(path):
        loaded.append(str(Path(path).resolve()))
        return np.full((4, 4, 3), .5, dtype=np.float32)
    from ue_framework.methods import s4_optimization as native
    with patch.object(native, 'load_image_rgb_float', fixture_image), patch.object(native, 'read_yolo_annotations', lambda _:[]):
        result = wiring.run(trainer, images, output/'labels', torch.ones(8, 256), output,
                            {'fixture':'explicit toy loss; not learned-Teacher validation'}, bounded=True)
    expected = [images[int(i)] for i in np.random.RandomState(0).permutation(20)[:16]]
    assert loaded == expected and trainer.observed_paths == [expected, expected]
    assert result['steps'] == 1 and result['schedule_complete'] is False
    assert not torch.equal(before, trainer.fourier_coeff)
    pack = torch.load(output/'global_params.pt', map_location='cpu', weights_only=True)
    state = torch.load(output/'native_s4/latest_continuation.pt', map_location='cpu', weights_only=False)
    assert tuple(pack['fourier_coeff'].shape) == (16, 3)
    assert tuple(state['trainer']['fourier_coeff'].shape) == (64, 3)
    assert state['cursor_state']['step'] == 1 and {'python','numpy','torch','cuda'} <= state['rng'].keys()
    assert 'num_classes' not in pack
    report = dict(state='passed', scope='component-only CPU validation', compiled_sources=len(changed),
        actual_canonical_optimizer_toy_Adam_updates=1, actual_toy_EOT_draws=2,
        original_full_list_seed0_first16_exact=True, retained_candidate_shape=[64,3],
        retained_final_shape=[16,3], old_B2_evidence_rejected=True, no_learned_Teacher_model=True,
        invalid_TFSS_proofs_rejected=rejected_proofs,
        actual_dataset_images_read=0, model_forwards=0, GPU_initialized=torch.cuda.is_initialized(),
        full_native_CPU_gate_passed=False, CUDA_gate_passed=False, formal_launch=False,
        cleanup_policy='retain_all')
    (output/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True, type=Path)
    main(p.parse_args().output)
