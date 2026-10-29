# COCO official-evaluation independent scientific review — 2026-10-29

- Reviewer: /root/coco_official_eval_scientific_review_20261029, independent delegated read-only context; parent conclusions were not used.
- Candidate: ed192435cd3bd30cbf412c731288dae472af43ce; diff base: 55ad8e80fdf5a24b676dc9b1ea117c3c0987b7e6.
- Result: scientifically_incorrect — do_not_run until B1 and B2 are closed.

## Blockers

1. **The paired metric absolute_clean_harm is declared but not emitted.** official_eval.json lists it; evaluation.py's paired_metrics returns P/U/S/WCD/TCD3 and metadata only. phase_official_eval stores that object without validating the configured key set. The later report's per-class clean-minus-S4 AP50 does not satisfy this official paired output.

2. **Val image bytes are not checked against the pinned per-image manifest.** runtime.py pins and hashes the manifest file, but coco_eval.py builds paths from annotations and checks existence and dimensions only. The manifest contains image_sha256 and image_bytes, but the inference path never reads them. A same-name, same-dimension substituted JPEG can reach COCOeval. The bounded smoke compares inference on those same current bytes and cannot establish image identity.

## Evidence gap

The immutable source release contains risk_estimator_policy and strict_Original_S4_equivalence_claim=false, and the official result links the release by SHA256. phase_spec/status/accepted do not include those values directly. They are recoverable via the release hash, so this is an attribution gap rather than a primary blocker.

## Checks that passed

- Candidate HEAD matched the packet commit, and the source-release hash matched the smoke phase-spec binding.
- Static review confirmed fixed Clean200/S4_200 checkpoint binding, matched initialization, sparse COCO80 mapping, pinned pycocotools, standard COCOeval maxDets, and no update/selection path.
- Packet smoke reports CUDA 0 for both checkpoints, 36 images each, finite raw pre-NMS outputs, exact native-reference matches, and 8 total raw forwards. This is bounded smoke evidence only; no official metrics were produced or inferred.
- Packet reports the CPU contract validator and compilation passed. I did not rerun tests, metrics, registration, or experiments.

This verdict applies only to the bounded official-evaluation phase. It does not approve future efficacy, causal, or final-report claims.
