# CUP classwise training-damage audit — 2026-10-15

CPU replay of76 recorded train-damage cases (two38-case sequences, EOT2 each), not76 independent images or official training. No valAP-based weighting. Actual GPU sink and5-step resume are separate evidence.

| class | calibration case-observations | replay EOT-observations | clipped damage EMA | final global relative weight |
|---|---:|---:|---:|---:|
|aeroplane|2|8|-1.3965905e-05|0.66555|
|bicycle|3|12|0.00026856155|1.22747|
|bird|2|8|0.00088107067|2|
|boat|2|8|0.00036112553|1.52443|
|bottle|3|12|0.00010292855|0.848474|
|bus|2|8|9.7950891e-05|0.83947|
|car|3|12|-1.1885076e-05|0.668334|
|cat|2|8|-0.00024508114|0.5|
|chair|7|28|0.00032829646|1.41065|
|cow|2|8|0.00043365686|1.81401|
|diningtable|2|8|9.8585576e-05|0.840611|
|dog|3|12|0.00012022058|0.880695|
|horse|2|8|0.00027177213|1.23659|
|motorbike|3|12|-0.00059718256|0.5|
|pottedplant|4|16|-0.00011323722|0.549162|
|sheep|2|8|-0.0003837342|0.5|
|sofa|3|12|-2.6058931e-05|0.649674|
|train|2|8|0.00044062683|1.84489|
|tvmonitor|5|20|-0.00019708681|0.5|

All19NTclasses observed; absent histories unchanged; actual bounded weight normalization passed. Final entropy=2.83082802655, relativeweights[0.5,2.0];11of76 scalar aggregations differ from uniform. This does not measure naturally rare-class AP. Person excluded from weights/history.

Actual GPU CUP10steps/20EOT, history fromzero;12batches show gradient change, no manual bootstrap/history. Data are fixed multiclass train queries chosen before probe; original whole-train query sampler remains formal policy.

No final per-class AP, victim effect or Person-to-Dog FP removal claim. Both formal runs and final single scientific review pending.

## Actual complete S3 audit — 2026-10-17
Final checkpoint CUP.steps=CNLP.completed_steps=Adam.steps=9144; all19 NT classes have positive recorded query-observation counts, Person history remains0. EMA history finite and within locked signed clip. This validates recorded history/lifecycle, not per-class efficacy. Final primary WCD=.2697248045357672 (bicycle), TCD3=.26075386351959123; S0 WCD=.2810323798542793 (dog), TCD3=.2578511679110964. Mixed class effects must be read in S2_S3_VOC20_PER_CLASS.csv; do not claim CUP eliminates Person-to-Dog false positives. Full checkpoint metadata in execution/s3_noise_completion_verification_20261017.json, actual metrics in s3_archive/raw/comparison.json.
