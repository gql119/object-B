"""Source-derived CPU audit of S4 retention decomposition, not an S5 trainer.

Executes the unchanged S4 scalar retention and ROI-pooling functions on
synthetic feature maps. No model, dataset, optimizer or remote work is run.
Real S4 disable/gradient/update equivalence remains a separate GPU gate.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/S5_CUP_S4"


def load_original(path, name):
    raw = path.read_bytes()
    source = raw.decode("utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.get_source_segment(source, node)
    namespace = {"torch": torch, "Tuple": Tuple}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[name], dict(path=str(path.relative_to(ROOT)), function=name,
                                line=node.lineno, source_sha256=hashlib.sha256(raw).hexdigest())


def run():
    retain, retain_source = load_original(ROOT / "ue_framework/methods/legacy_kproto_ret.py",
                                          "non_target_retention_loss")
    pool, pool_source = load_original(ROOT / "ue_framework/methods/alce_losses.py", "masked_prototype")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rows, cases = [], []
    definitions = {
        "unequal_counts": [(0, (0, 0, 5, 5)), (0, (3, 1, 8, 6)), (0, (1, 5, 6, 9)),
                           (11, (6, 6, 10, 10))],
        "cross_and_same_class_overlap": [(0, (0, 0, 8, 8)), (0, (2, 2, 10, 10)),
                                         (11, (1, 1, 9, 9)), (2, (5, 5, 10, 10))],
        "empty_and_subpixel_roi": [(0, (0, 0, 0, 0)), (11, (4, 4, 8, 8))],
        "all_19_non_target_classes": [(c, (c % 5, c % 4, c % 5 + 4, c % 4 + 5))
                                       for c in range(20) if c != 14],
        "no_non_target_roi": [],
    }
    for name, specs in definitions.items():
        clean = torch.randn(1, 8, 12, 12) * .05
        clean.requires_grad_(True)
        poison = (clean.detach() + torch.randn_like(clean) * .01).requires_grad_(True)
        pc, pp, labels = [], [], []
        invalid = 0
        for cls, (x1, y1, x2, y2) in specs:
            mask = torch.zeros(1, 1, 12, 12)
            mask[:, :, y1:y2, x1:x2] = 1
            a, va = pool(clean.detach(), mask, min_pixels=1.0)
            p, vp = pool(poison, mask, min_pixels=1.0)
            assert torch.equal(va, vp)
            if bool(va.item()):
                pc.append(a[0]); pp.append(p[0]); labels.append(cls)
            else:
                invalid += 1
        # Exercise an actual sub-feature-pixel mask; its mass is below 1.
        if name == "empty_and_subpixel_roi":
            mask = torch.full((1, 1, 12, 12), .001)
            _, valid = pool(poison, mask, min_pixels=1.0)
            assert not bool(valid.item())
            invalid += 1
        ntp = torch.stack(pp) if pp else poison.new_empty((0, poison.shape[1]))
        ntc = torch.stack(pc) if pc else clean.new_empty((0, clean.shape[1]))
        original = retain(ntp, ntc) if pp else poison.sum() * 0
        if pp:
            instance = (ntp - ntc.detach()).square().sum(-1)
            class_losses, pis = [], []
            for cls in sorted(set(labels)):
                idx = torch.tensor([i for i, c in enumerate(labels) if c == cls])
                value = instance.index_select(0, idx).mean()
                pi = len(idx) / len(labels)
                class_losses.append(value); pis.append(pi)
                rows.append(dict(case=name, cls=cls, valid_rois=len(idx), total_valid_rois=len(labels),
                                 pi=pi, L_ret_class=float(value.detach()),
                                 baseline_contribution=float((pi * value).detach()), input_kind="synthetic_CPU"))
            pi_t = original.new_tensor(pis)
            values = torch.stack(class_losses)
            decomposed = (pi_t * values).sum()
            uniform_class_mean = values.mean()
            # Proposed equivalent correction form; baseline w=pi is exact.
            baseline_residual_form = original + ((pi_t - pi_t) * values).sum()
            g0 = torch.autograd.grad(original, poison, retain_graph=True)[0]
            g1 = torch.autograd.grad(decomposed, poison, retain_graph=True)[0]
            g2 = torch.autograd.grad(baseline_residual_form, poison, retain_graph=True)[0]
            clean_grad = torch.autograd.grad(original, clean, allow_unused=True)[0]
            assert clean_grad is None
            scalar_error = float((original - decomposed).detach().abs())
            gradient_error = float((g0 - g1).abs().max())
            noop_error = max(float((original - baseline_residual_form).detach().abs()), float((g0-g2).abs().max()))
            assert scalar_error <= 1e-7 and gradient_error <= 1e-7 and noop_error == 0
            result = dict(name=name, valid_rois=len(labels), absent_non_target_classes=19-len(set(labels)),
                          invalid_rois=invalid, scalar_absolute_difference=scalar_error,
                          gradient_max_absolute_difference=gradient_error,
                          baseline_correction_noop_max_difference=noop_error,
                          pi_sum=sum(pis), naive_equal_class_mean_difference=float((original-uniform_class_mean).detach().abs()),
                          clean_reference_gradient="detached", finite=True)
        else:
            gradient = torch.autograd.grad(original, poison)[0]
            assert original == 0 and not bool(gradient.any())
            result = dict(name=name, valid_rois=0, absent_non_target_classes=19, invalid_rois=invalid,
                          scalar_absolute_difference=0., gradient_max_absolute_difference=0.,
                          baseline_correction_noop_max_difference=0., pi_sum=0.,
                          clean_reference_gradient="detached", finite=True)
        cases.append(result)
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(schema="s5.source-derived-retention-cpu-audit.v1", status="PASS_CPU_SYNTHETIC_ONLY",
                  torch_version=torch.__version__, device="cpu", seed=0,
                  sources=[retain_source, pool_source], cases=cases, model_forwards=0,
                  optimizer_updates=0, real_TRAIN32_equivalence="NOT_RUN",
                  canonical_S4_loss_files_modified=False,
                  finding="instance-count-weighted class decomposition preserves the existing retention definition; equal class weighting generally does not")
    (OUT / "S5_RETENTION_CPU_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (OUT / "CUP_CLASSWISE_RETENTION_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
