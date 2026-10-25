# COCO paired official evaluation and final report implementation

Recorded 2026-10-25. This is production source preparation and CPU contract evidence only. It is not an official COCO result, GPU validation, scientific review, or phase release.

## Paired official evaluation

`coco_s4.phase_official_eval` consumes only the accepted `clean200` and `s4_200` native `last.pt@200` checkpoints. It rechecks the serial receipts, checkpoint/full-state hashes, zero validation forwards, and identical pretraining-state digest. It evaluates each fixed checkpoint against the same original 5,000-image val2017 annotation file with the pinned isolated COCOeval 2.0.10 addon. The prediction path checks all pre-NMS tensors for finiteness, retains native decoded predictions, and uses the frozen 640/batch36/FP32/confidence/NMS/max-detection settings.

For each model it records all 80 categories' AP50, AP75, AP50:95, Recall50 and Precision50, plus official metric JSON, per-class CSV and prediction provenance. It derives signed P/U/S, WCD and TCD3 from the matched Clean/S4 outputs without clipping or checkpoint selection. Parameter updates, method-statistic extraction and metric-driven tuning are prohibited. The smoke entrypoint evaluates one bounded batch per checkpoint with official metrics and artifact ingestion disabled, compares raw and decoded predictions to an independent native `YOLO.predict` model instance, retains all artifacts, and checks a total raw-forward budget of 100.

The isolated validation runner is `python -B -m coco_s4.official_eval_release prepare|launch-smoke|register --tag TAG`; registration additionally needs one independent, blocker-free review of that exact released source/config and actual smoke. The runner checks the existing serial slot, all accepted predecessor receipts, current capacity, remote smoke bytes and binding before uploading one `official_eval_READY.json`. It cannot create or start another formal controller.

## Final report

`coco_s4.phase_final_report` is deterministic CPU postprocessing of accepted Teacher/Clean/S4 official metrics and export diagnostics. It writes a JSON report, an 80-class CSV, and a Markdown summary. It includes the Original S4 protocol identity, one-seed descriptive P/U/S/WCD/TCD3, per-class AP/recall/precision and absolute clean harm, shared VOC class names, and the accepted decoded-export support/budget/PSNR audit. LPIPS remains `null` with state `not_validated`; no value is inferred. It does not make a causal or significance claim and cannot update or select model parameters.

This derived-report step needs no model GPU pass or independent experiment review. Its CPU contract and exact accepted receipts must pass before the existing serial controller receives the final-report READY record. Its release helper must reuse the already accepted official-evaluation source root and can only create/upload the bound report release and READY receipt; there is no second scheduler.

## Validation actually run

`python -B -m py_compile` succeeded for both phase wrappers, both release helpers, and the contract validator. `python -B -m coco_s4.validate_post_eval_contracts` passed: four altered official-evaluation contracts were rejected; synthetic signed paired-metric checks and the 80-row report derivation passed; all 20 shared VOC names aligned. The report explicitly carried unmeasured LPIPS as null. The test opened zero dataset images, ran zero GPU units, generated zero official metrics, and performed no formal registration.

## Open gates

The current formal Teacher200 remains the earlier immutable runtime. These new source files do not change it. No official-evaluation or final-report release, smoke, scientific review, READY receipt, or formal stage has been created. The official-evaluation stage still requires accepted predecessors, a fresh capacity receipt, a current immutable release, actual bounded GPU smoke, one independent review and main-agent closure. Final report cannot run until that official evaluation and all earlier stages are accepted.
