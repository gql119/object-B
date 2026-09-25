# Experiment B: full person-box perturbation

This repository contains the source for the bbox-half experiment: a shared Fourier perturbation is applied across each person bounding box, with half strength where a non-target box overlaps. The configured pixel budget is 16/255.

## Entry points

- Noise optimization: `scripts/run_full_noise_bbox_half_20260925.sh`
- Poisoned-data materialization: `scripts/run_materialization_bbox_half_20260925.sh`
- Randomly initialized YOLOv8n victim training and clean VOC evaluation: `scripts/run_victim_bbox_half_20260925.sh`
- Experiment configuration: `ue_framework/configs/exp_voc_person_legacy_kproto_ret_bbox_half.yaml`

Before running, set the dataset root, surrogate checkpoint, output root, and Python executable in the configuration or environment-specific launch settings. The dataset, checkpoints, training outputs, and local review artifacts are intentionally not included.
