# COCO export25k validation — in progress

Protocol: `coco80-ntgt-native-tal-then-gt-risk-v1`, with dynamic selected P5
coordinates. The NT-risk boundary extension is not globally equivalent to
strict Original S4. No S4-Full is used.

The existing controller accepted Fourier24 at 2026-10-28 03:55 UTC. Its receipt
SHA256 is `4c6852a2ccac25f540034b619b1c7a5651ef2daffcf2862656fbe6d5b2754220`;
it records 20448 optimizer updates, 24 epochs and 852 updates per epoch. The
learned parameter SHA256 is
`a0afe86edc6d7b843673cb420ce7ade991e4e60a797b92af63d44ffe0b036691`.
At 05:52 UTC the same live controller was waiting for the export phase release,
with no active scientific child. These are completed Fourier results, not
victim AP or a completed poisoned-dataset export.

The actual export dependency reader passed: Teacher checkpoint/full state,
Teacher evaluation, calibration/K8/P5 selected39/all79/P3 reference artifacts,
learned Fourier parameters and immutable controller binding all matched.
Nothing was recalibrated or retrained.

This candidate adds an isolated one-image CPU execution of the existing export
entrypoint before its one-image CUDA execution. CPU-only RNG isolation remains
at the existing adapter boundary. The CUDA/formal computation, frozen S4 core,
native risk mathematics, labels, data split, loss and 8/4/0 projection are
unchanged. The final-generation native composition is compared numerically
against the saved-parameter generator/export construction on the same image.

The GPU/export release must hash-bind a successful same-source CPU export
receipt with the same Teacher, phase configuration and learned Fourier
receipt/parameters. Missing or mismatched CPU evidence cannot register a
formal export. CPU cannot masquerade as CUDA or write production READY.

Measured local checks: export composition/calibration/schedule contracts passed
(exclusive8, overlap4, outside0; zero data images/model forwards); one valid
simulated CPU evidence consumer and seven negative cases passed. Changed
modules parsed successfully and `git diff --check` passed. These are component
checks; real CPU/CUDA export probes and the new export scientific review are
still pending. No formal export has been launched by this document.

Next: publish this candidate, execute the retained real-Teacher CPU probe,
then current-source CUDA probe, perform exactly one independent export
scientific review and main closure, and publish READY for the existing sole
controller. All probe artifacts remain retained; no official metrics/ingest,
parameter updates or cleanup are allowed. AT parallel-resource testing is
deferred by the latest direct human instruction; no AT probe was launched.

The first real CPU probe (`coco_export25k_20261028_cpu01`, source `cb028b4`)
failed before output/model execution: the old export function imported
`DATA_ROOT` from `phase_contract`, which does not export that name. The error
and failed status are retained. This candidate imports the same unchanged
constant from its canonical `runtime` module and checks every production
relative import locally. It does not substitute a new dataset or relax any
scientific check. The failed ID is not reused; a new CPU probe is required.
