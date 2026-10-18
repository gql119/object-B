"""Paired P3/P5 retention observations; original S4 remains the reference."""
import torch

from .alce_losses import masked_prototype
from .s4_tfss import S4Trainer
from .s5_retention_core import S5FrozenRetention


class RetentionMeasurements:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s5_ret_layer = 'P3'
        self.s5_decomposition_noop = True
        self.s5_feature_phase = None
        self.s5_heads = {}
        self.s5_model_shapes = {}
        self.s5_output_hook = self.surrogate.register_forward_hook(self._s5_head_hook)

    def _s5_head_hook(self, module, inputs, output):
        if self.s5_feature_phase is not None:
            raw = output[1] if isinstance(output, tuple) else output
            if not isinstance(raw, dict) or not {'scores', 'boxes', 'feats'} <= raw.keys():
                raise ValueError('S5 diagnostic requires actual native raw detector outputs')
            self.s5_heads[self.s5_feature_phase] = raw

    def _feature(self, image, grad):
        self.s5_feature_phase = 'poison' if grad else 'clean'
        try:
            result = super()._feature(image, grad)
            for name in ('model.15', 'model.21'):
                value = self.multi_features[name]
                self.s5_model_shapes[name] = list(value.shape)
            return result
        finally:
            self.s5_feature_phase = None

    def _batch_loss(self, items, prototypes):
        self.s5_retention_records = {'P3': [], 'P5': []}
        self.s5_heads = {}
        result = super()._batch_loss(items, prototypes)
        if self.s5_model_shapes['model.15'][-1] != 4*self.s5_model_shapes['model.21'][-1]:
            raise ValueError('Runtime P3/P5 output geometry is not the expected stride8/32 pair')
        return result

    def s5_record_retention(self, layer, clean, poison, masks, annotations, eligible, index, shape):
        annotations = [a for a in annotations if a.get('bbox') is not None]
        if len(masks) != len(annotations):
            raise ValueError('GT/ROI order differs before validity filtering')
        for ordinal, (mask, ann) in enumerate(zip(masks, annotations)):
            cls = int(ann['cls'])
            if cls == self.target_class_id or not 0 <= cls < 20:
                raise ValueError('Unexpected target/invalid class in retention ROI')
            c, cv = masked_prototype(clean, mask, min_pixels=1.)
            p, pv = masked_prototype(poison, mask, min_pixels=1.)
            if not torch.equal(cv, pv):
                raise ValueError('Clean/poison ROI validity mismatch')
            valid = bool(cv.item())
            loss = (p[0]-c[0].detach()).square().sum() if valid else None
            if loss is not None and not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classwise retention contribution')
            self.s5_retention_records[layer].append(dict(image_index=index, gt_non_target_ordinal=ordinal,
                cls=cls, eligible_image=bool(eligible), pool_valid=valid,
                included=bool(eligible and valid), mask_mass=float(mask.sum()),
                original_shape=list(shape), bbox=list(ann['bbox']), channels=poison.shape[1],
                feature_shape=list(poison.shape[-2:]), loss=loss))

    def s5_reduce_retention(self, original, poison):
        if self.orth_feature_layer != 'model.15':
            raise ValueError('This diagnostic must start from the audited actual S4 P3 path')
        self.s5_ret_losses = {}
        self.s5_class_losses = {}
        self.s5_class_pi = {}
        for layer in ('P3', 'P5'):
            rows = [r for r in self.s5_retention_records[layer] if r['included']]
            losses = [r['loss'] for r in rows]
            mean = torch.stack(losses).mean() if losses else poison.sum()*0
            self.s5_ret_losses[layer] = mean
            self.s5_class_losses[layer] = {}
            self.s5_class_pi[layer] = {}
            for cls in sorted({r['cls'] for r in rows}):
                values = [r['loss'] for r in rows if r['cls'] == cls]
                self.s5_class_losses[layer][cls] = torch.stack(values).mean()
                self.s5_class_pi[layer][cls] = len(values)/len(rows)
        error = float((original-self.s5_ret_losses['P3']).detach().abs())
        if error > 1e-7:
            raise ValueError('Observed P3 GT mapping fails to reconstruct real S4 retention: '+str(error))
        self.s5_original_ret = original
        self.s5_original_mapping_error = error
        p3_ids = {(r['image_index'],r['gt_non_target_ordinal']) for r in self.s5_retention_records['P3'] if r['included']}
        p5_ids = {(r['image_index'],r['gt_non_target_ordinal']) for r in self.s5_retention_records['P5'] if r['included']}
        self.s5_missing_P5_rois = sorted(p3_ids-p5_ids)
        self.s5_additional_P5_rois = sorted(p5_ids-p3_ids)
        base = original if self.s5_ret_layer == 'P3' else self.s5_ret_losses[self.s5_ret_layer]
        # These are layer probes, not CUP. Baseline factors w=1 recover base exactly.
        if self.s5_decomposition_noop:
            values = self.s5_class_losses[self.s5_ret_layer]
            pi = self.s5_class_pi[self.s5_ret_layer]
            zero = torch.stack([base.new_tensor(pi[k])*(base.new_tensor(1.)-1.)*v
                                for k,v in values.items()]).sum() if values else base*0
            return base + zero
        return base

    def s5_close_observer(self):
        self.s5_output_hook.remove()


class S5LayerTrainer(RetentionMeasurements, S4Trainer, S5FrozenRetention):
    """MRO keeps S4 spatial/Adaptive wrappers and swaps only Frozen retention."""
