# Original S4 COCO Fourier24 phase implementation

This is production-wrapper implementation and CPU contract evidence only. It is
not a calibration result, Fourier run, GPU smoke, scientific review, formal
registration, or Experiment A outcome.

The new `coco_s4.phase_fourier` entrypoint uses the existing Original S4
`B2SemanticTrainer` and native Fourier optimizer without changing their loss,
EOT, adaptive inner-step, or coefficient-update code. Its wrapper binds the
fixed Teacher `last@200`, exact accepted calibration receipt, 25K train
manifest, K=8 prototype cache, StrictP5/SafeBG artifacts, source release, native
dependency hashes, and COCO data hashes. It imports the accepted calibration's
consumer artifacts into the new phase output by hash and runs the existing
semantic-identity guard against that isolated exact copy.

The original trainer collects all images containing Person and skips images
whose original support has at most 10 pixels inside the optimization loop. On
COCO, that makes the update count differ from the required dataset-derived
schedule. This wrapper supplies only the previously accepted calibration
manifest of Person images whose original `bbox_half_overlap` support exceeds 10
pixels. Formal acceptance requires exactly `24 * ceil(N / 16)` actual Adam
updates, with `N` taken from that frozen manifest, 24 updates-per-epoch counts,
and no dropped batch. A mismatch fails the phase and retains its evidence.

The isolated smoke path runs one eligible clean-train image through the same
production loss and optimizer entrypoint, with one update and EOT=2. It requires
finite full loss/delta/Fourier gradient, a changed Fourier coefficient, no
change to the frozen Teacher, and perturbation confined to the original support.
Official metrics and artifact ingestion remain disabled. The bounded candidate
smoke and formal stage use the existing serial controller; no second workflow
or READY registration is created by the release helper.

Neither the CPU full-loss probe nor GPU smoke has been prepared or run. The CPU
probe now enters the same Original S4 `train_universal` loss/optimizer path on
exactly one accepted support-eligible COCO train image, selects CPU before any
CUDA availability checks, retains a separate CPU-only RNG snapshot, and cannot
be combined with the GPU smoke or formal mode. It is intentionally deferred
until accepted Teacher evaluation and calibration artifacts exist. A contract
test verifies both CPU and GPU probe selectors restrict the trainer to one
eligible image; it does not open that image or claim loss-chain execution.

The GPU smoke gates require the current workflow's
accepted Teacher200, separate read-only fixed-last evaluation, accepted real
COCO calibration and strict24D artifacts, current committed/published source,
and fresh server capacity. One current-release GPU smoke, one independent
scientific review, and main-agent closure remain required before registering
`fourier24_READY.json` for the existing serial controller.

CPU validation performed here: 15 phase-contract rejection cases, 13 frozen
Original S4 method-config rejection cases, and 13 eligible-manifest/bounded
probe-selection cases (including duplicate, unsorted, wrong-step, escaped-path,
and CPU/GPU probe-selection checks). Module imports passed. The tests opened no
COCO images, initialized no GPU, produced no official metrics, and created no
workflow registration. The current Teacher was observed separately as healthy
and remains on its immutable existing runtime; this implementation does not
alter it.

Native Fourier code does not save its Adam optimizer state. This wrapper stores
full before/after Python, NumPy, Torch CPU, and CUDA RNG snapshots and keeps all
outputs, but it does not claim exact worker/prefetch or mid-epoch resumption.
Failures remain in their own retained output; recovery requires a fresh unique
identity after diagnosing the actual exit and process state.
