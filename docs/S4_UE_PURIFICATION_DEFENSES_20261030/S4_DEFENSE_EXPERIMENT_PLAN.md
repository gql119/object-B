# Experiment plan: OP, AVATAR, D-VAE, and LE on Original S4 VOC20

Date: 2026-10-30
Planning status: preparation only; no method-specific RunSpecs, purification outputs, or formal runs exist.
**No duration estimate is available:** no VOC method path was executed or benchmarked.

## Frozen parent and paired evaluation protocol

The parent dataset is accepted Original S4 VOC20 with Person id 14 and all 20 GT classes retained. The existing matched victim protocol is random-YAML YOLOv8n, seed 0, 200 epochs, 640 input, batch 36, workers 16, and the existing locked optimizer/schedule/augmentation/AMP settings. Train split contains 16,551 images; clean VOC validation contains 4,952 images and must remain byte-identical and unpurified. The primary victim checkpoint is `last.pt@200`; `best.pt` is secondary only.

For each defense D, the future pair must be `CleanTrain → D → random-init victim` and `S4PoisonTrain → D → matched random-init victim`, followed by the same untouched clean validation. Clean and Poison must use the same predeclared defense model, strength, data policy, cache semantics, and victim protocol unless the original method explicitly requires different handling; any such difference must be documented before results. No validation/test image or label may train a purifier, choose its strength, or select the final checkpoint.

Each run requires a unique immutable defense/model/source/config identity and separate output root. The transform manifest must bind input/output image hashes, stable image IDs, dimensions, label-byte hashes, decoded class/box hashes, purifier checkpoint/source, and parameters. The common contract in `defenses/common/pair_contract.py` enforces only ID/dimension/GT invariants; actual image decoding, semantic preservation, method correctness, and detector behavior need separate method tests.

## Method gates and future stages

| Method | Future execution entry | Purification stages and current blockers | Checkpoint/data/cache policy | RunSpec and resource estimate |
|---|---|---|---|---|
| OP | Not implemented. Candidate may only be named after a detector projection definition is approved and coded. | Define a multi-instance direction estimator and projection application without collapsing each image to one label. Validate class co-occurrence, Person-only, Person-free, small objects, background, and projection rank before generating a dataset. | Reuse only accepted clean/Poison train inputs. Any learned direction model gets its own SHA/manifest. Cache must be keyed by exact input SHA, method source, config, and classifier/direction checkpoint. | No Clean/Poison RunSpec. CPU/CUDA resources and duration `UNKNOWN`; do not extrapolate CIFAR timings. |
| AVATAR | Not implemented. A wrapper must load a verified model/checkpoint for the declared data distribution/resolution and record sampler/timestep. | Pretrained diffusion sanitization followed by paired image/GT audit. Establish 640-space input/output rules, denoising strength selection independent of final clean validation, and small-object/boundary semantic checks. | No VOC checkpoint currently staged. Do not substitute CIFAR or ImageNet model without a predeclared adaptation protocol. Cache by checkpoint and input/config hashes; keep clean val untouched. | No RunSpec. Model loading, memory, per-image time, total preprocessing time, and victim time `UNKNOWN`. |
| D-VAE | Not implemented. A future entry must preserve both trained stages and their separate checkpoint identities. | Define class/instance/spatial conditioning for all VOC GT, including empty frames and overlaps; predeclare rate/KLD targets and reconstruction selection without val leakage; then audit both outputs at original geometry. | No compatible VOC two-stage weights. If training from scratch is proposed, specify legal training split and exact data/optimizer/rate config before any run; do not use one global class label. Cache stage 1 and stage 2 separately with full lineage. | No RunSpec. Two-stage VAE training/inference and memory/time are `UNKNOWN`. |
| LE | Not implemented. A future entry must trace both diffusion condition and image-guidance condition separately. | Lock the authorized auxiliary unprotected dataset and overlap policy, then define class-condition handling for multi-instance images and preserve the pixel/perceptual guidance mechanism. The complete VOC val/test split cannot supply training data or parameter selection. | CIFAR-only checkpoint links do not satisfy VOC. No checkpoint was downloaded. Do not use placeholder `xxx.pt`; cache only exact approved model/data/input/config identities. | No RunSpec. Auxiliary-model training, inference, data storage, and victim cost are `UNKNOWN`. |

No current method has a valid way to populate a formal `Clean` or `Poison` RunSpec. Once a method-specific science definition, model/data sources, CPU image tests, native bounded CUDA validation, independent review, main closure, and immutable READY gates all pass, the existing serial controller may be used only under an explicit launch authorization. The present task did not authorize four full 200-epoch runs.

## Required preflight before any future method can be called ready

1. Verify source commit/license, checkpoint existence/hash/provenance, data origin/overlap, model architecture, and runtime dependencies.
2. Freeze method-specific parameters from the primary paper/author config or mark missing fields `NOT_DISCLOSED`; do not guess.
3. CPU test real samples for Person+non-target co-occurrence, Person-only, no Person, and small object, with exact sample/label identity and original geometry.
4. Verify every input is readable; output is finite, in range, encoded/decoded consistently, and sample-to-label mapping remains exact. Add a separate semantic/localization test; metadata equality alone is insufficient.
5. Run matched Clean/Poison current-source CUDA probes only after the active-GPU resource gate passes. Probes must use the real model, resolution, method code, and full GT; no synthetic model or batch fallback.
6. Obtain exactly one method-appropriate independent scientific review and main closure after evidence exists; freeze paired RunSpecs and release before formal execution.
7. Train the Clean and Poison victims serially from matched fresh random initialization, evaluate both on untouched clean VOC val4952, and publish manifests/checkpoint hashes/results without overwriting prior artifacts.

## Failure, recovery, and results

No method-specific restart/recovery implementation is present. A future run must preserve the original RunID, console/logs, intermediate purifier checkpoints, RNG/data-loader state, exact source/config, output manifest, and terminal reason. A resumed job must verify its native continuation boundary; if safe resume is not proven, retain the failure and use a new immutable RunID only after new authorization. Do not overwrite the accepted S4 dataset, original AT checkpoints, or previous reports.

Report at least Person AP50, 19-class NT mAP50, all-class mAP50, NT AP75, Person/NT AP50:95, per-class AP50, WCD, and TCD3 for both paired victims, all from `last.pt@200` as primary. Report each defense's Person recovery and absolute NT percentage-point change, including Clean degradation. For matching Clean and Poison values define `P_D = 1 - AP50_person(Poison_D)/AP50_person(Clean_D)`, `U_D = mAP50_NT(Poison_D)/mAP50_NT(Clean_D)`, and `S_D = 2 P_D U_D/(P_D+U_D)`. A zero/invalid denominator or undefined sum is `N/A`/null; do not replace it with zero. Existing clean-ST and S4-Poison-ST reference values do not replace the defense-specific Clean baseline.

All current method result fields are unavailable. No runtime, GPU-memory, preprocessing-throughput, or training-duration figure is inferred from the classifier papers or from unrelated Original S4 runs.
