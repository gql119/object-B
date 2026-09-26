# Experiment B: full person-box perturbation

This repository contains the source for the bbox-half experiment: a shared Fourier perturbation is applied across each person bounding box, with half strength where a non-target box overlaps. The configured pixel budget is 16/255.

## Entry points

- Noise optimization: `scripts/run_full_noise_bbox_half_20260925.sh`
- Poisoned-data materialization: `scripts/run_materialization_bbox_half_20260925.sh`
- Randomly initialized YOLOv8n victim training and clean VOC evaluation: `scripts/run_victim_bbox_half_20260925.sh`
- Experiment configuration: `ue_framework/configs/exp_voc_person_legacy_kproto_ret_bbox_half.yaml`

Before running, set the dataset root, surrogate checkpoint, output root, and Python executable in the configuration or environment-specific launch settings. The dataset, checkpoints, training outputs, and local review artifacts are intentionally not included.

## B2-T semantic variant (2026-09-26)

The B2-T implementation and its completed single-seed run are recorded in [`experiments/b2_t_20260926/`](experiments/b2_t_20260926/README.md). It uses a 48-of-64 TSFL channel selection with specificity weight 1.0 and `lambda_orth=0`; the result package includes clean VOC AP50/recall, per-class metrics, hashes, full-data preflight summary, and the 200-epoch training CSV. B2-T source/config additions are under `ue_framework/methods/b2_semantic.py`, `ue_framework/methods/semantic_coordinates.py`, and `ue_framework/configs/exp_voc_person_b2_t.yaml` / `exp_voc_person_b2_to.yaml`. Dataset, checkpoints, weights, teacher artifacts, and absolute server paths are excluded.
