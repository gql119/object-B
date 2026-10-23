"""S5 retention-layer diagnostic core, statically derived from S4.

Only two observation sites and the Frozen L_ret reduction sink differ from
SemanticFrozenTrainer._batch_loss. The original file is not modified.
See docs/S5_CUP_S4/S5_RETENTION_CORE_PARENT.diff. No CNLP or CUP is active here.
"""
import torch
import torch.nn.functional as F
from .b2_semantic import SemanticFrozenTrainer
from .legacy_kproto_ret import _rectangle, select_prototype, kproto_losses, non_target_retention_loss
from .semantic_coordinates import normalized_orth, CLASSES
from .orth_reference_design import reference_energy_orth
from .safebg import safe_background_direction


class S5FrozenRetention(SemanticFrozenTrainer):
    def _batch_loss(self, items, prototypes):
        """One surrogate clean/poisoned forward for a legacy-style padded batch."""
        pad_h = (max(image.shape[0] for image, _, _ in items) + 31) // 32 * 32
        pad_w = (max(image.shape[1] for image, _, _ in items) + 31) // 32 * 32
        images, supports = [], []
        for image, _, support in items:
            h, w = image.shape[:2]
            images.append(F.pad(torch.from_numpy(image).permute(2, 0, 1).float(),
                                (0, pad_w - w, 0, pad_h - h), value=0.5))
            supports.append(F.pad(torch.from_numpy(support)[None].float(),
                                  (0, pad_w - w, 0, pad_h - h)))
        tensor = torch.stack(images).to(self.device)
        support_t = torch.stack(supports).to(self.device)
        raw, delta, poison, _, _ = self._compose_delta_batched(
            tensor, support_t, torch.zeros_like(support_t), self._active_coords,
            self.fourier_coeff[self._active_idx], self.suppress_small, current_epoch=self._epoch
        )
        clean_aug, poison_aug = self._apply_shared_eot_pair_batched(tensor, poison)
        clean = self._feature(clean_aug, grad=False).detach()
        orth_clean = self.multi_features["model.15"].detach() if self.orth_feature_layer else None
        poisoned = self._feature(poison_aug, grad=True)
        orth_poisoned = self.multi_features["model.15"] if self.orth_feature_layer else None
        image_losses, non_target_pairs = [], []
        selected_losses, raw_selected_losses, displacements, clean_targets, overlap_weights = [], [], [], [], []
        orth_displacements, orth_clean_targets, orth_overlap_weights = [], [], []
        safe_bg_diagnostics = []
        clean_raw_bg_cosines, clean_safe_bg_cosines = [], []
        class_drifts = {c: [] for c in self.bases}
        target_drift, proto_ids, proto_similarity, context_norm = [], [], [], []
        for b, (image, annotations, support) in enumerate(items):
            h0_map, hp_map = clean[b:b + 1], poisoned[b:b + 1]
            target_anns = [a for a in annotations if int(a["cls"]) == self.target_class_id]
            nt_anns = [a for a in annotations if int(a["cls"]) != self.target_class_id]
            if self.orth_feature_layer:
                orth_h0_map = orth_clean[b:b + 1]
                orth_hp_map = orth_poisoned[b:b + 1]
                orth_targets, orth_contexts, orth_non_targets = self._masks(
                    annotations, support, image.shape[:2], orth_hp_map, (pad_h, pad_w))
                orth_valid = False
                for ann, target_mask, context_mask in zip(target_anns, orth_targets, orth_contexts):
                    h0_orth = self._pooled(orth_h0_map, [target_mask])
                    hp_orth = self._pooled(orth_hp_map, [target_mask])
                    ct_orth = self._pooled(orth_h0_map, [context_mask])
                    if h0_orth and hp_orth and ct_orth:
                        orth_valid = True
                        orth_clean_targets.append(h0_orth[0])
                        orth_displacements.append(hp_orth[0] - h0_orth[0])
                        box = _rectangle(image.shape[:2], ann["bbox"])
                        overlap = any(bool((box * _rectangle(image.shape[:2], a["bbox"])).any()) for a in nt_anns)
                        orth_overlap_weights.append(self.overlap_weight if overlap else 1.)
                orth_nt0 = self._pooled(orth_h0_map, orth_non_targets)
                orth_ntp = self._pooled(orth_hp_map, orth_non_targets)
                if orth_valid and orth_nt0 and orth_ntp:
                    non_target_pairs.append((torch.stack(orth_ntp), torch.stack(orth_nt0)))
                self.s5_record_retention('P3', orth_h0_map, orth_hp_map,
                    orth_non_targets, nt_anns, orth_valid, b, image.shape[:2])
            targets, contexts, non_targets = self._masks(
                annotations, support, image.shape[:2], hp_map, (pad_h, pad_w)
            )
            self.s5_record_retention('P5', h0_map, hp_map, non_targets,
                nt_anns, orth_valid, b, image.shape[:2])
            h0, hp, ct = [], [], []
            overlaps = []
            for ann, target_mask, context_mask in zip(target_anns, targets, contexts):
                clean_target = self._pooled(h0_map, [target_mask])
                poison_target = self._pooled(hp_map, [target_mask])
                context = self._pooled(h0_map, [context_mask])
                if clean_target and poison_target and context:
                    h0.append(clean_target[0])
                    hp.append(poison_target[0])
                    ct.append(context[0])
                    box = _rectangle(image.shape[:2], ann["bbox"])
                    overlap = any(bool((box * _rectangle(image.shape[:2], a["bbox"])).any()) for a in nt_anns)
                    overlaps.append(self.overlap_weight if overlap else 1.)
            if not hp:
                continue
            h0, hp, ct = torch.stack(h0), torch.stack(hp), torch.stack(ct)
            indices, selected, similarity = select_prototype(ct, prototypes)
            nt0 = self._pooled(h0_map, non_targets)
            ntp = self._pooled(hp_map, non_targets)
            nt0 = torch.stack(nt0) if nt0 else hp_map.new_zeros((0, hp_map.shape[1]))
            ntp = torch.stack(ntp) if ntp else hp_map.new_zeros((0, hp_map.shape[1]))
            image_losses.append(kproto_losses(hp, h0, selected, ntp, nt0, self.margin))
            selected_clean = h0.index_select(1, self.channels)
            selected_poison = hp.index_select(1, self.channels)
            selected_raw_bg = selected.index_select(1, self.channels)
            if self.safe_bg_enabled:
                raw_selected_losses.append(kproto_losses(selected_poison, selected_clean,
                    selected_raw_bg, ntp, nt0, self.margin))
                safe_bg, safe_diag = safe_background_direction(
                    selected_clean.detach(), selected_raw_bg.detach(), self.safe_bases,
                    active_k=self.safe_bg_active_k,
                    projector_rtol=self.safe_bg_projector_rtol, eps=self.orth_eps,
                    mode=self.safe_bg_mode, safe_rank=self.safe_bg_rank,
                    triplet_cache=self.safe_bg_cache,
                    removal_alpha=self.safe_bg_removal_alpha,
                )
                selected_losses.append(kproto_losses(selected_poison, selected_clean,
                    safe_bg.detach(), ntp, nt0, self.margin))
                safe_bg_diagnostics.append(safe_diag)
                clean_raw_bg_cosines.append(F.cosine_similarity(selected_clean, selected_raw_bg, dim=-1).mean())
                clean_safe_bg_cosines.append(F.cosine_similarity(selected_clean, safe_bg, dim=-1).mean())
            else:
                selected_losses.append(kproto_losses(selected_poison, selected_clean,
                    selected_raw_bg, ntp, nt0, self.margin))
            displacements.append(hp - h0.detach())
            clean_targets.append(h0.detach())
            overlap_weights.extend(overlaps)
            for ann, mask in zip(nt_anns, non_targets):
                a = self._pooled(h0_map, [mask]); p = self._pooled(hp_map, [mask])
                if a and p:
                    class_drifts[int(ann["cls"])].append((p[0]-a[0]).square().sum())
            if ntp.numel() and not self.orth_feature_layer:
                non_target_pairs.append((ntp, nt0))
            target_drift.append((hp - h0).square().sum(dim=-1).mean())
            proto_ids.extend(indices.detach().cpu().tolist())
            proto_similarity.append(similarity.mean())
            context_norm.append(ct.norm(dim=-1).mean())
        if not image_losses:
            return None
        bg, rel, _, cos_clean, cos_bg = [
            torch.stack([loss[i] for loss in image_losses]).mean() for i in range(5)
        ]
        if non_target_pairs:
            ret = non_target_retention_loss(
                torch.cat([pair[0] for pair in non_target_pairs]),
                torch.cat([pair[1] for pair in non_target_pairs]),
            )
        else:
            ret = poisoned.sum() * 0.0
        ret = self.s5_reduce_retention(ret, poisoned)
        bg_full, rel_full, cos_clean_full, cos_bg_full = bg, rel, cos_clean, cos_bg
        if raw_selected_losses:
            bg_raw, rel_raw, _, _, cos_raw_bg = [
                torch.stack([loss[i] for loss in raw_selected_losses]).mean() for i in range(5)
            ]
        else:
            bg_raw, rel_raw, cos_raw_bg = bg, rel, cos_bg
        bg, rel, _, cos_clean, cos_bg = [torch.stack([x[i] for x in selected_losses]).mean() for i in range(5)]
        displacement = torch.cat(displacements)
        selected_displacement = displacement.index_select(1, self.channels)
        orth_source = torch.stack(orth_displacements) if self.orth_feature_layer else displacement
        orth_weights = (orth_overlap_weights if self.orth_feature_layer else overlap_weights)
        orth_clean_source = (torch.stack(orth_clean_targets) if self.orth_feature_layer
                             else torch.cat(clean_targets))
        orth_displacement = orth_source.index_select(1, self.orth_channels)
        orth, conflicts, worst = normalized_orth(orth_displacement, self.orth_bases,
            orth_source.new_tensor(orth_weights), self.worst_k, self.orth_eps)
        orth_full, _, _ = normalized_orth(orth_source, self.orth_full_bases,
            orth_source.new_tensor(orth_weights), self.worst_k, self.orth_eps)
        reference_orth, _, _ = reference_energy_orth(orth_displacement,
            orth_clean_source.index_select(1, self.orth_channels), self.orth_bases,
            orth_source.new_tensor(orth_weights), self.worst_k, self.orth_eps)
        orth_objective = reference_orth if self.orth_objective == 'reference_energy' else orth
        tv = self._tv_loss(raw)
        budget = F.relu(raw.abs().max() - self.eps)
        weighted_bg = self.lambda_bg * bg
        weighted_rel = self.lambda_rel * rel
        weighted_ret = self.lambda_ret * ret
        total = weighted_bg + weighted_rel + weighted_ret + self.lambda_orth * orth_objective + self.lambda_tv * tv + self.lambda_budget * budget
        self._semantic_terms = {"BG_sel": weighted_bg, "rel_sel": weighted_rel,
                                "ret": weighted_ret, "orth": self.lambda_orth * orth_objective}
        self._orth_raw = orth_objective
        self._frozen_total = total
        metrics = {
            "L_BG": bg, "L_rel": rel, "L_ret": ret,
            "weighted_L_BG": weighted_bg, "weighted_L_rel": weighted_rel,
            "weighted_L_ret": weighted_ret, "L_tv": tv, "L_budget": budget,
            "L_total": total, "cos_target_clean": cos_clean, "cos_target_bg": cos_bg,
            "relative_gap": cos_bg - cos_clean, "non_target_feature_drift": ret,
            "target_feature_drift": torch.stack(target_drift).mean(),
            "selected_proto_id": torch.tensor(float(proto_ids[0]), device=self.device),
            "selected_proto_ids": ",".join(str(int(i)) for i in proto_ids),
            "context_proto_similarity": torch.stack(proto_similarity).mean(),
            "context_feature_norm": torch.stack(context_norm).mean(),
            "max_abs_delta": delta.abs().max(), "mean_abs_delta": delta.abs().mean(),
            "saturation_ratio": (delta.abs() >= self.eps - 1e-6).float().mean(),
            "support_area_ratio": (support_t > 0).float().mean(),
            "outside_support_max": (delta * (support_t == 0)).abs().max(),
        }
        metrics.update({
            "L_BG_safe": bg, "L_rel_safe": rel,
            "L_BG_full": bg_full, "L_rel_full": rel_full,
            "cos_target_bg_full": cos_bg_full, "cos_target_clean_full": cos_clean_full,
            "L_BG_sel": bg, "L_rel_sel": rel, "cos_target_bg_sel": cos_bg,
            "cos_target_clean_sel": cos_clean, "L_orth_worst": orth,
            "L_orth_worst_tsfl_channels": orth,
            "L_orth_worst_full_channels": orth_full,
            "L_orth_reference_energy": reference_orth,
            "L_orth_objective": orth_objective,
            "orth_objective_reference_energy": poisoned.new_tensor(float(self.orth_objective == 'reference_energy')),
            "weighted_L_orth": self.lambda_orth * orth_objective,
            "target_displacement_norm": displacement.norm(dim=-1).mean(),
            "target_displacement_tsfl_norm": selected_displacement.norm(dim=-1).mean(),
            "orth_coordinate_channels": poisoned.new_tensor(float(self.orth_channels.numel())),
            "mean_non_target_feature_drift": ret,
        })
        if self.safe_bg_enabled:
            retentions = torch.cat([d["retention"] for d in safe_bg_diagnostics])
            bg_norms = torch.cat([d["d_bg_norm"] for d in safe_bg_diagnostics])
            safe_norms = torch.cat([d["d_safe_norm"] for d in safe_bg_diagnostics])
            removed_norms = torch.cat([d["removed_norm"] for d in safe_bg_diagnostics])
            removed_ratios = torch.cat([d["removed_ratio"] for d in safe_bg_diagnostics])
            orth_cosines = torch.cat([d["orth_cosine"] for d in safe_bg_diagnostics])
            active_ratios = torch.cat([d["active_projection_ratios"] for d in safe_bg_diagnostics])
            active_classes = [row for d in safe_bg_diagnostics for row in d["active_classes"]]
            metrics.update({
                "L_BG_safe_lr": bg if self.safe_bg_mode == "low_rank" else poisoned.new_zeros(()),
                "L_rel_safe_lr": rel if self.safe_bg_mode == "low_rank" else poisoned.new_zeros(()),
                "L_BG_raw": bg_raw, "L_rel_raw": rel_raw,
                "cos_poison_raw_bg": cos_raw_bg, "cos_poison_safe_bg": cos_bg,
                "cos_clean_raw_bg": torch.stack(clean_raw_bg_cosines).mean(),
                "cos_clean_safe_bg": torch.stack(clean_safe_bg_cosines).mean(),
                "bg_direction_norm": bg_norms.mean(), "safe_direction_norm": safe_norms.mean(),
                "safe_direction_scaled_norm": torch.cat([d["d_safe_scaled_norm"] for d in safe_bg_diagnostics]).mean(),
                "safe_direction_retention": retentions.median(),
                "safe_direction_retention_raw": retentions.median(),
                "safe_retention_q10": torch.quantile(retentions, 0.10),
                "safe_retention_q25": torch.quantile(retentions, 0.25),
                "safe_retention_q75": torch.quantile(retentions, 0.75),
                "safe_retention_q90": torch.quantile(retentions, 0.90),
                "removed_nt_component_norm": removed_norms.mean(),
                "removed_nt_component_ratio": removed_ratios.mean(),
                "removed_component_ratio": removed_ratios.mean(),
                "projected_component_norm": torch.cat([d["projected_component_norm"] for d in safe_bg_diagnostics]).mean(),
                "removed_component_norm": removed_norms.mean(),
                "retained_projected_component_norm": torch.cat([d["retained_projected_component_norm"] for d in safe_bg_diagnostics]).mean(),
                "effective_removal_ratio": removed_ratios.mean(),
                "safe_bg_removal_alpha": poisoned.new_tensor(self.safe_bg_removal_alpha),
                "collapsed_safe_direction_count": torch.cat([d["collapsed"] for d in safe_bg_diagnostics]).float().sum(),
                "collapsed_safe_direction_ratio": torch.cat([d["collapsed"] for d in safe_bg_diagnostics]).float().mean(),
                "cos_d_safe_projected_nt": orth_cosines.mean(),
                "decomposition_error": torch.cat([d["decomposition_error"] for d in safe_bg_diagnostics]).max(),
                "safe_projector_rank": torch.cat([d["projector_rank"] for d in safe_bg_diagnostics]).float().mean(),
                "safe_projector_basis_columns": torch.cat([d["projector_basis_columns"] for d in safe_bg_diagnostics]).float().mean(),
                "safe_projector_orthonormal_error": torch.cat([d["orthonormal_error"] for d in safe_bg_diagnostics]).max(),
                "safe_projector_top8_energy_ratio": torch.cat([d["top8_energy_ratio"] for d in safe_bg_diagnostics]).mean(),
                "safe_bg_rank_configured": poisoned.new_tensor(float(self.safe_bg_rank or 0)),
                "safe_bg_feature_layer_id": poisoned.new_tensor(21.),
                "orth_feature_layer_id": poisoned.new_tensor(15.),
                "adaptive_feature_layer_id": poisoned.new_tensor(15.),
                "strict_tsfl_total_channels": poisoned.new_tensor(self.strict_summary["total_channels"]),
                "strict_tsfl_target_candidate_count": poisoned.new_tensor(self.strict_summary["target_candidates"]),
                "strict_tsfl_safe_candidate_count": poisoned.new_tensor(self.strict_summary["safe_candidates"]),
                "strict_tsfl_final_selected_count": poisoned.new_tensor(self.strict_summary["final_selected_channels"]),
                "mean_target_sensitivity_selected": poisoned.new_tensor(self.strict_summary["mean_target_sensitivity_selected_raw"]),
                "mean_worst_nt_sensitivity_selected": poisoned.new_tensor(self.strict_summary["mean_S_nt_worst_selected"]),
                "mean_target_sensitivity_unselected": poisoned.new_tensor(self.strict_summary["mean_target_sensitivity_unselected_raw"]),
                "mean_worst_nt_sensitivity_unselected": poisoned.new_tensor(self.strict_summary["mean_S_nt_worst_unselected"]),
                "target_to_nt_ratio_selected": poisoned.new_tensor(self.strict_summary["mean_ratio_selected"]),
            })
            class_frequency = {}
            spectrum = torch.cat([d["singular_values_top16"] for d in safe_bg_diagnostics]).mean(0)
            for j, value in enumerate(spectrum):
                metrics[f"safe_projector_singular_{j + 1}"] = value
            for rank in range(self.safe_bg_active_k):
                values = [row[rank] for row in active_classes]
                class_frequency[rank] = max(sorted(set(values)), key=lambda c: values.count(c))
                metrics[f"active_nt_class_{rank + 1}"] = poisoned.new_tensor(class_frequency[rank])
                metrics[f"projection_ratio_class_{rank + 1}"] = active_ratios[:, rank].mean()
        for j, c in enumerate(sorted(self.bases)):
            metrics["L_orth_"+CLASSES[c]] = conflicts[j]
            metrics["feature_drift_"+CLASSES[c]] = (torch.stack(class_drifts[c]).mean()
                if class_drifts[c] else poisoned.sum()*0.)
            metrics["feature_count_"+CLASSES[c]] = poisoned.new_tensor(len(class_drifts[c]))
        for j in range(self.worst_k):
            c = worst[j]
            metrics[f"worst_class_{j+1}"] = poisoned.new_tensor(c)
            metrics[f"worst_Lorth_{j+1}"] = conflicts[sorted(self.bases).index(c)]
        for key in ("num_selected_channels", "selected_channel_fraction", "mean_S_target_selected",
                    "mean_S_nt_selected", "mean_S_target_unselected", "mean_S_nt_unselected",
                    "specificity_selected_mean", "channel_selection_overlap_with_previous_run"):
            source_key = "num_channels_selected" if key == "num_selected_channels" else key
            value = len(self.channels)/len(self.tsfl["S_target"]) if key == "selected_channel_fraction" else self.summary[source_key]
            metrics[key] = poisoned.new_tensor(value)
        return total, metrics
