# Original S4 VOC20 AT-4 / AT-8

This package extends only Victim training. Scientific generation parent is
`b1e29510e7b1ed5fa0fe361a27132d8790e65c14`; native Victim/AMP base is
`ab626917f20917e58cf30033ac99df00938cb561`. All inherited `ue_framework` bytes
are locked. No S4-Full or COCO NT-risk extension is imported.

The immutable static Original-S4 data retain Person-exclusive 8/255,
Person/NT-overlap 4/255, outside-Person 0. AT is a separate dynamic defense
over the **whole augmented, LetterBoxed, normalized input**. Project-defined
PGD10 uses (epsilon, step)=(4,1)/255 or (8,2)/255, uniform random start, native
full-GT box/cls/DFL loss ascent, and [0,1] clipping. These details are not
attributed to undisclosed MI-UE Table4 parameters.

`ATTrainer.preprocess_batch` calls native preprocessing before PGD. PGD uses
`autograd.grad` only for input gradients, detaches every iteration, freezes BN
running-stat updates, restores every module flag/parameter requires_grad, and
preserves existing accumulated parameter gradients. Outer AMP, GradScaler,
SGD, clipping, EMA, scheduler and warmup use the pinned native methods. Native
TAL is unchanged: every transformed GT is submitted, unmatched GT counts are
recorded, and a nonempty batch with no native positive supervision fails.

The PGD generator is independent of the outer RNG. Exact L-inf endpoints are
computed in float64 before casting inward to FP32; no epsilon tolerance is
added. A failed PGD is never replaced with random noise.

Formal configuration remains 200 epochs, random YAML YOLOv8n, seed0,
batch36, workers16, imgsz640, SGD, cosine LR and close_mosaic100. Default native
Mosaic/HSV/flip parameters and all resolved train defaults are recorded in
`configs/PROTOCOL.json`. OOM batch reduction is rejected. CPU fixtures use
batch2/imgsz160/workers0/FP32 and are explicitly **not** formal configuration
equivalence or CUDA readiness evidence.

## Local CPU checks

Use the existing project Python environment, UTF8 and `-B`; install nothing.

```
python -B -m unittest discover -s tests -p test_s4_at.py -v
python -B -m robustness_s4_at.validate_cpu --fixtures execution/fixtures --out execution/UNIQUE_CPU_TAG
python -B -m robustness_s4_at.worker --run-id S4AT4_Clean_seed0 --fixture-root execution/fixtures --out execution/UNIQUE_LOOP_TAG --device cpu
```

Fixtures are exactly hash-checked public VOC training records; the protected
fixture uses the accepted materialization manifest. No validation image enters
PGD or a bounded training fixture. The fixture YAML has a train-only alias for
native loader construction; bounded runs abort before any validator forward.

## Deferred CUDA preflight and formal sequence

Current COCO formal work has priority. Do not launch an AT controller, GPU smoke
or training concurrently. After that workflow ends, use the existing RRCTL
mechanism for current published-source bounded native **batch36/imgsz640**
Clean/Poison PGD10/AMP/update validation, then exactly one scientific review and
main-agent closure. Gate receipts must contain real current-source evidence.
No reviewer is called before actual CUDA evidence.

Only then run the one serial queue:

```
python -B -m robustness_s4_at.queue --preflight ACCEPTED_PREFLIGHT.json --out OWNED_UNIQUE_OUTPUT --run
```

Order: Clean-AT4 -> Poison-AT4 -> Clean-AT8 -> Poison-AT8. There is no automatic
retry after a failed arm. The queue checks RAM32GiB, free GPU24000MiB, disk150GiB,
three IOwait windows, other live COCO controller, and own GPU processes. Other
users' processes are never altered. Queue code is dormant until explicitly
launched through RRCTL after gates pass; it is not a second active scheduler.

## Checkpoints, resume and clean evaluation

Each run has separate outputs. Native last/best pointers update within that run;
completed epoch copies, full FP32 model/optimizer/scaler/scheduler/EMA/RNG/PGD
states and all failures are retained without deletion. Same-run resume requires
exact source/config/data/budget identity, a completed epoch with cleared
accumulated gradients, and absence of later completed results. Cross-arm and
finished-run resume are rejected. The controlled CPU next-update replay can be
exact; workers16/prefetch continuation is **not claimed bitwise equivalent**.
CUDA/native-loader resume evidence remains a formal preflight gate.

Training validation and the separate last-primary/best-secondary evaluations
use untouched VOC4952 and full20 classes. Plain native validation has no PGD.
The aggregate uses same-budget Clean-AT denominators, keeps all per-class
AP50/AP75/AP50:95/P/R, reports absolute percentage-point changes, WCD/TCD3 and
raw P/U/S. Undefined denominators remain null with reasons. Signed ratios are
not clipped; values use fractional units before percentage presentation.
Clean-ST/S4-ST references are separate provenance-checked rows.

The AT progress writer updates `audit/progress.json` after every outer batch.
Future RRCTL RunSpecs must declare that relative path as both progress and
artifact, with a validated short local pull destination. No active COCO binding
or source is modified to add this telemetry.
