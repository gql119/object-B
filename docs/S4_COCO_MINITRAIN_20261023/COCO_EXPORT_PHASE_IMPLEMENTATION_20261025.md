# Original S4 COCO poisoned-train export phase

This is a production phase implementation for the existing serial Experiment A
controller. It is not an export result, GPU validation, phase registration, or
formal launch. The active Teacher runtime remains immutable while this source is
developed.

The phase is bound to the accepted fixed EMA `last@200` Teacher, the accepted
COCO train-only calibration, and the accepted 24-epoch Fourier result. Before
the export can be prepared, its release reader checks the live controller is
waiting for `export25k`, all preceding receipts and artifacts match their
controller-bound hashes, the Fourier step count equals
`24 * ceil(eligible Person images / 16)`, and a fresh capacity observation
passes the existing admission guards. The independent worker scope and all
outputs use one unique candidate under the existing owned `releases/` and
`validation/` roots. No second scheduler is created.

The formal exporter reads only the fixed 25K clean training manifest. It
preserves every label byte and all non-Person image bytes. For Person images it
reuses the Original S4 generator and clean-only per-image TFSS-U map cache,
applies `W = 1 - .15 * Rnt * (1 - Qt)` with `eta=.85`, then uses the original
lossless PNG projection. Every written PNG is decoded and checked against the
8/4/0 byte budget; every source image and label is matched to the accepted
manifest. It records image hashes, support areas, decoded pixel deltas, PSNR,
and TFSS map evidence. Validation is never opened and official metrics and
artifact ingestion stay disabled.

The isolated current-candidate smoke is limited to one accepted Person training
image. It reaches the same production export path, decodes the resulting PNG,
and checks labels and budgets without official metrics, artifact ingestion, or
parameter updates. Formal registration remains blocked until local CPU
contracts, a current committed actual CUDA smoke, exactly one independent
scientific review, and main-agent closure all pass.

Local checks completed for this source: `py_compile`, the new synthetic
TFSS/PNG budget and dependency-contract test, remote dependency-probe syntax,
CLI import, and `git diff --check`. These checks read zero real COCO images and
initialize no GPU. No export, smoke, reviewer, or Experiment A phase was run.
