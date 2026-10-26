# One-review finding F01: nearest-candidate arithmetic

The single independent review of candidate3524258 returned
scientifically_incorrect / do_not_run. The original negative JSON/Markdown are
retained unchanged in execution/COCO_RISK_INDEPENDENT_SCIENTIFIC_REVIEW_20261027.*.
No complete25K audit or formal optimization has started.

F01 identifies an implementation error inside the Level C candidate's declared
float64 nearest-location rule. The old implementation derived GT center/size in
float32 before promotion. With actual float32 endpoints constructed from
[7.99,3.99,8.010001,4.01] and anchors [[4,4],[12,4]], it rounded the x center to8.0
and falsely tied the two anchors, choosing index0. Deriving from the same endpoints
in float64 gives center8.000000476837158 and selects index1.

The repair promotes endpoints before subtraction/addition, then computes the
existing size-normalized squared-distance/argmin rule in float64. It retains
native-index tie breaking, validity guards, no epsilon, unchanged GT and the
existing geometric mask when nonempty. No estimator formula, class averaging,
rank, threshold, loss, Qt/W definition or8/4/0 budget changes. This precision
repair is Level A within an independently disclosed Level C risk extension;
it does not make that extension equivalent to strict Original S4.

The new targeted validator tests the exact actual helper against an independent
Python-float64 oracle, the review counterexample, a true center tie, invalid boxes
and nonempty-geometric-mask preservation. Current-commit CPU and CUDA production
statistics probes remain required before main-agent closure. No second or closing
reviewer will be called. The earlier real backpack used five geometric candidates;
the finding does not retroactively invalidate those recorded geometric-case
measurements. Its population frequency is still unknown.

Packet raw-file SHA256:18498ca8440fa42e5ad18306e378a609bb5894e55759fc3b7e01b82a7229c3bb.
The readiness tool reports a different, canonical-JSON SHA256:
ed1451e4e5007c34654176143ab148cae473250758d08f8cbe9b0b89a378047e.
Both were independently recomputed; the distinction is serialization, not drift.
