# S4 four-defense readiness audit

Date: 2026-10-30
Source parent: `gql119/object-B` snapshot `64daca67003b3eab2321c7bc22775a90a95d9842`; independent adaptation branch `codex/s4-ue-purification-defenses-20261030`.
Scope: preparation only; no defense model was loaded and no formal run was launched.

## Audited project path

The existing robustness implementation is on the source-bound evaluation branch at commit `0c1082f8190e10c6b0581a3f5dfa3f5170e5e5e2`:

- `ue_framework/s4_robustness.py`, blob `aa4123492a135e736c724478fc5c81815c18b857`, exposes `transform_rgb()` for `none`, `data_aug`, JPEG, grayscale, bit-depth reduction, Gaussian/mean/median filtering. It accepts original-resolution HWC `uint8` RGB. Unknown transform identifiers raise an error. There is no OP/AVATAR/D-VAE/LE transform, learned purifier, checkpoint loader, or method-specific config here.
- `scripts/run_s4_robustness.py`, blob `1eb148d853a3d76243e5e97bd704a635bb83a1a0`, drives the serial Clean/Poison defense pipeline, materializes image views, checks image/label identity, trains the source-bound YOLO victim, and evaluates the untouched clean validation view. Its transform data flow does not instantiate the four upstream methods.
- `docs/S4_ROBUSTNESS_FINAL_20261022/S4_ROBUSTNESS_LOCK.json`, blob `12f8ffdec4b55cc79388cd3d97d79f9e1ed1aef8`, freezes the existing S4 victim setup: random YAML initialization, 200 epochs, 640 image size, batch 36, workers 16, SGD, cosine schedule, native augmentation, AMP, seed 0, 16,551 train images and clean validation of 4,952 images. This lock is evidence for the existing pipeline only; it is not a frozen configuration for any of the four new methods.

The adaptation worktree is isolated from the F: research checkout and the active AT worktree. The active AT runtime source remains pinned to `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`; no Original S4, AT trainer, RunSpec, data, or active release file was edited. The latest accessible state file at `2026-10-30 09:16:36 +08:00` says `FORMAL_CLEAN_AT4_RUNNING_WAITING_FOR_NATURAL_BOUNDARY`, arm CleanAT4 at epoch 68 / 67 complete, 30,977 batches, 15,922 updates, 10 legal AMP skips, and no results. Two read-only RRCTL health attempts in this continuation returned `worker_output: worker emitted invalid JSON during health`; therefore this is a state-file observation, **not a fresh server confirmation**. No AT process was touched.

## Per-method gates and status

| Method | Status | Existing project implementation / detector adapter | Frozen method config and usable checkpoint | Data and GT | CPU/CUDA evidence | Formal Clean/Poison launch |
|---|---|---|---|---|---|---|
| OP | `SCIENTIFIC_ADAPTATION_REQUIRED` | None. Upstream is class-labelled image classification; no detector entry. | No VOC projection rank, aggregation, or detector checkpoint. | Base S4 train/clean data exist; no principled class/instance direction mapping. | Shared metadata contract unit tests pass; OP not executed. CUDA not run. | No; no RunSpecs or READY. |
| AVATAR | `BLOCKED_DEPENDENCY` | None. Existing image transforms do not call diffusion. | CIFAR-10 32×32 and ImageNet 256×256 source configs; no verified VOC20/640 checkpoint or frozen timestep. | Base S4 data exist; no geometry/semantic validation for a VOC-compatible model. | Shared metadata contract unit tests pass; AVATAR not executed. CUDA not run. | No; checkpoint/model gate is open. |
| D-VAE | `SCIENTIFIC_ADAPTATION_REQUIRED` | None. Upstream source is classification-oriented. | CIFAR/ImageNet-mini examples only; no VOC20 two-stage model/rate-target config or checkpoint. | Base S4 data exist; 20-way multi-instance condition and rate selection remain undefined. | Shared metadata contract unit tests pass; D-VAE not executed. CUDA not run. | No; no detection definition or READY. |
| LE | `BLOCKED_DEPENDENCY` | None. No conditional diffusion wrapper in the project path. | CIFAR model links/config only; no VOC20/640 model or pair of frozen configs. | Extra unprotected model-training data/access policy is not approved. No use of clean validation/test. | Shared metadata contract unit tests pass; LE not executed. CUDA not run. | No; data/checkpoint/threat-model gates are open. |

### Common pair-contract work

This branch adds `defenses/common/pair_contract.py`. It checks that input and output records retain the same sample-ID set, dimensions, label-byte SHA, and decoded VOC20 `xyxy` GT while allowing image pixels to change. It rejects duplicate IDs, invalid hashes, out-of-range classes/coordinates, empty-area boxes, missing samples, and changed labels/geometry. This is a metadata contract for pixel-only purification, not a detector wrapper. It does not inspect image bytes, establish visual semantics, run inference, or prove that object boundaries remain recognizable.

Ten standard-library pair-contract unit cases and one readiness-registry guard pass (11 tests total). They use synthetic metadata records to exercise Person+non-target coexistence, Person-only, Person-free, small boxes, and explicit failure modes. They are **not real-image or method tests**. No actual VOC image path test could be credited because none of the four purifier/checkpoint paths is available. Thus per-method `cpu_method_test=false`, CUDA probes are all `false`, and `READY_VERIFIED` is not assigned.

## End-to-end readiness checklist

| Gate | OP | AVATAR | D-VAE | LE |
|---|---|---|---|---|
| Upstream commit and source entry audited | Yes | Yes | Yes | Yes |
| Current project method implementation | No | No | No | No |
| YOLOv8/GT-preserving adapter | No | No | No | No |
| Per-method immutable config / Clean-Poison RunSpecs | No | No | No | No |
| Verified method checkpoint/model file | No | No | No | No |
| Approved external training data, if required | N/A pending adaptation | N/A for pretrained model; checkpoint absent | N/A pending adaptation/training protocol | No |
| Clean/Poison input mapping and manifest | Existing generic pipeline only | Existing generic pipeline only | Existing generic pipeline only | Existing generic pipeline only |
| Real-image co-occurrence / Person-only / no-Person / small-object test | No | No | No | No |
| Clean validation unchanged proof for this method | No | No | No | No |
| One method review / main closure / actual READY | No | No | No | No |

The generic robustness pipeline has paired Clean/Poison support and a clean validation path. That does not make the four methods ready: the method-specific sanitizer must first produce auditable paired views without label/sample drift, then pass actual image and semantic checks, then use two immutable matching RunSpecs and independent phase gates.

## Decision

No method is `READY_VERIFIED`, `PARTIALLY_IMPLEMENTED`, or safe to start. OP and D-VAE require new detection definitions; AVATAR and LE are blocked by missing compatible model/data packages, and LE also has an unresolved threat-model condition. A dedicated contract unit test is the only execution evidence from this branch. The GPU readiness flag remains false; no full defense/Victim training was authorized or started.
