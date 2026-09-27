# B2-TO: TSFL with angular orthogonality regularization

## Outcome

B2-TO completed the 24-epoch noise run and a 200-epoch randomly initialized YOLOv8n victim evaluation on the byte-identical clean VOC validation split. Person AP50 was 0.04154 (4.15%), non-target macro AP50 was 0.69464 (69.46%), and person recall at the evaluator operating point was 0.00.

Compared with B2, person AP50 increased by 1.95 percentage points, non-target macro AP50 increased by 0.03 points, and overall AP50 increased by 0.13 points. Compared with B2-T, person AP50 increased by 0.85 points, non-target AP50 by 0.41 points, and overall AP50 by 0.44 points. The target metric moved opposite to intended suppression relative to both references; this single run does not support an effectiveness claim.

## Protocol and provenance

- Noise optimization: 24 epochs, 9,144 steps, 3.6305 hours; formal noise validation succeeded.
- Materialized training set: 16,551 images. The victim entry point completed its built-in full-data validation before training; validation used 4,952 clean VOC images copied byte-for-byte.
- Victim: YOLOv8n, random initialization (`pretrained=false`), seed 0, SGD, 200 epochs, batch 36, 640-pixel images; `best.pt` selected.
- Method: 48/64 selected semantic channels, fraction 0.75, specificity 1.0, angular orthogonality weight 0.0005; adaptive batch 1, inner LR 1e-4, lambda_adapt 1.0, lambda_nt 0.25. Fourier learning rate was 0.001.
- Source method commit: `aeb88b075a3e40446f96d982637d44cad5fe5a62`. Recovery supervisor commit: `7906bc5fe4338eb40774412bb838b6ecefddd06a`. The first pipeline supervisor stopped before downstream stages because of an import-path error; recovery reused the verified noise checkpoint (`noise_retrained=false`) and then completed materialization, full-data validation, and victim training.
- Configuration SHA256: `
95747041ad671b716b0cffa66af68fe434f94281a841f3a0bb376d5be7cbb24a
`. Noise `global_params.pt` SHA256: `4ae28d877b41c457e121bcc9597671133f789bfe6e484d4691d958ec073c1714`.
- Compact raw evidence is included here: run spec, victim/noise status, clean metrics, results CSV, recovery status and stage exit codes. No dataset or checkpoint is included.

## Interpretation

These are descriptive results from one seed, not a causal estimate. Fourier LR differs from B2/B2-T, so this is not an isolated orthogonality ablation. AP50 change does not demonstrate causal background absorption. Person AP50 is nonzero despite zero reported person recall at this evaluator operating point.

## Per-class AP50

| Class | B2 | B2-T | B2-TO | TO - B2 (pp) |
|---|---:|---:|---:|---:|
| aeroplane | 0.8400 | 0.8363 | 0.8385 | -0.15 |
| bicycle | 0.6460 | 0.6635 | 0.6485 | +0.25 |
| bird | 0.6510 | 0.6518 | 0.6676 | +1.66 |
| boat | 0.6865 | 0.6711 | 0.6675 | -1.90 |
| bottle | 0.4754 | 0.4713 | 0.5120 | +3.66 |
| bus | 0.8327 | 0.8103 | 0.8270 | -0.57 |
| car | 0.8866 | 0.8813 | 0.8799 | -0.67 |
| cat | 0.8093 | 0.8039 | 0.8045 | -0.48 |
| chair | 0.5325 | 0.5245 | 0.5404 | +0.79 |
| cow | 0.6876 | 0.6822 | 0.6804 | -0.71 |
| diningtable | 0.6807 | 0.6795 | 0.6780 | -0.26 |
| dog | 0.5875 | 0.5750 | 0.5834 | -0.41 |
| horse | 0.7237 | 0.7317 | 0.7330 | +0.93 |
| motorbike | 0.7201 | 0.7048 | 0.6978 | -2.23 |
| person | 0.0220 | 0.0330 | 0.0415 | +1.95 |
| pottedplant | 0.4417 | 0.4494 | 0.4291 | -1.26 |
| sheep | 0.7196 | 0.7173 | 0.7235 | +0.39 |
| sofa | 0.7180 | 0.7037 | 0.6985 | -1.95 |
| train | 0.8347 | 0.8274 | 0.8476 | +1.29 |
| tvmonitor | 0.7188 | 0.7345 | 0.7409 | +2.21 |

Target/person AP50 is 0.04154; recall is 0.00. Largest class AP50 decreases vs B2:
motorbike (-2.23 pp), sofa (-1.95 pp), boat (-1.90 pp)
.


The committed YAML is a sanitized, single-seed template. Runtime dataset, run, teacher-checkpoint, and semantic-artifact paths were redacted; its SHA256 is recorded separately from the exact runtime config hash in metrics.json and provenance.json.
