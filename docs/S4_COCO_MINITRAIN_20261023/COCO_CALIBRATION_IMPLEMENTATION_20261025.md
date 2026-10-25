# Original S4 COCO calibration phase implementation

## Current candidate update (2026-10-25)

Teacher200 and its fixed-last full 5K assessment are now independently accepted;
the existing controller waits for calibration. This candidate loads the native
EMA DetectionModel (rather than its Sequential child), converts serialized FP16
to the canonical FP32 calibration model, and freezes it. A synthetic native CPU
probe matched canonical YOLO loading, raw outputs, and input gradients exactly.
The bounded calibration smoke now measures actual raw tensor CUDA device/index
and compares every P3/P5 sensitivity sum against independent original per-layer
GT-evidence loops on the same frozen Teacher and one real clean-train image.
The new reference function passed a separate synthetic CPU probe (640, Person0
and one NT class, 960 compared values, two reference forwards). These are local
checks; real current-candidate CUDA, one new calibration scientific review, and
formal registration remain open. The short RRCTL pull path, POSIX dependency
paths, and previous-receipt summary field are explicit transport/binding fixes.
No Original S4 math, thresholds, data split, checkpoint bytes, or training
hyperparameters changed. Historical implementation notes below are retained.

## State

This is a local, CPU-contract-validated implementation candidate for the
clean-train calibration phase. It is not a calibration result, a formal phase
release, a GPU smoke, or scientific-review evidence. The current Teacher200
workflow remains on its immutable runtime and is inspected only.

## Implemented path

`coco_s4.phase_calibration` binds to the accepted fixed EMA `last@200` Teacher,
the accepted preceding read-only 5K Teacher assessment, the exact Original S4
COCO configuration, the accepted 25K per-image manifest, and the current
immutable release. It verifies the actual image and label SHA for every train
image during statistics collection. The code retains raw P3/P5 GT-gradient
statistics, class feature pools, the original clean no-grad K=8 prototype bank,
and generated coordinate/reference artifacts. It runs the unchanged StrictP5
and SafeBG consumers and blocks when the actual selected channel space cannot
satisfy the frozen 24D protocol. Validation images and Teacher assessment AP
values are never used for statistics or method decisions.

`coco_s4.calibration_release` is a single-candidate release and bounded smoke
launcher for the existing serial controller. It requires the controller to be
waiting at `calibration`, both previous phases to be accepted, exact receipt
and checkpoint hashes, a fresh capacity receipt, a clean published source
commit, and absent remote candidate paths. It can launch only a one-image
production-path CUDA smoke in the owned validation root; it does not register a
formal phase or create another controller. A future scientific closure and
explicit registration route are still required before formal calibration.

The smoke is train-only and metric-free, limited to one eligible Person image
and at most 100 measured Teacher forwards. It retains all outputs. Formal
calibration is bound to the existing serial output and checks the single
current-candidate smoke/review closure before it can write `accepted.json`.

## Validation and limits

Local validation passed: the calibration phase contract validator rejected 17
invalid protocol, route, source-config, or per-file-hash cases; Python AST
parsing passed for all four affected Python modules; the existing serial
workflow `RunSpec` parsed successfully. The validator opened zero dataset
images, used zero GPU units, and created no formal registration. These checks
do not establish actual COCO statistics, checkpoint compatibility, CUDA loss
readiness, selected 24D compatibility, or any scientific result.

At 2026-10-24T23:55:05Z, a separate read-only RRCTL observation verified that
the existing Teacher200 was healthy at epoch 133 (epoch CSV rows through 132),
with 46,784 actual optimizer updates, 23 legal AMP skips, finite model/EMA/
optimizer state, and matching live controller/child identities. No change was
made to that run.

The calibration release preparation was deliberately not run: the serial
workflow has not accepted Teacher200 and `teacher_eval`, so the calibration
dependency and capacity gates are not open. The later GPU smoke, one
independent scientific review, main-agent closure, and formal registration
remain required. No official COCO metric or LPIPS value is reported here.
