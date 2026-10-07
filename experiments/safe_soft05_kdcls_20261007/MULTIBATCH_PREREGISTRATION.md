# KD-only multi-batch scale, fixed before rerunning sanity

The earlier five-step candidate was blocked because its utility/attack gradient
ratio reached 13.9756 (>10). This new calibration is limited to the added KD
term; the inherited `L_ret`, its input/features/reduction, `lambda_ret=0.25`,
`tau=2`, `beta=1`, and all Safe-Soft05 scientific settings remain unchanged.

Before reading any new five-step result, select the first **64** sorted VOC
training target images that also have at least one non-target GT. Split those
images into **four disjoint 16-image batches**. For each batch, run the frozen
parent forward/one-step Adaptive computation without changing the perturbation,
using clean-query positions **0, 4, 8, 12** in that order. There are exactly
**16 query measurements** and one existing shared EOT draw per measurement.
Keep all 16 measurements, including a zero-KD measurement if teacher TAL
assigns no non-target positive unit. No validation image or final AP is used.

Use the exact completed Safe-Soft05 background prototype cache, SHA256
`12889ba0f80b29c194a59bd9eb2a6d8b137e9aa4977fdb94ad84444628b316a7`;
verify its cache key against the current train data/model protocol. Compute

`kd_fixed_scale = arithmetic_mean(L_ret_raw across 16) /
                  arithmetic_mean(L_KD_raw across 16)`.

Require both means positive and finite, all teacher/student correspondence and
person exclusion checks to pass, and unchanged noise parameters. Save all
per-query values and the final scale to a hashed JSON artifact. The official
candidate reads only that artifact; it never recomputes the scale from the first
optimization step. This changes no `L_ret` calculation or coefficient.

After binding the artifact, use new isolated RunIDs for a no-update fixed-batch
probe and a five-step production-entrypoint smoke. The original gate remains:
all finite, correct assignment and pixel budget, and utility/attack gradient
ratio **at most 10 on every measured outer step**. If the gate fails, stop
without adjusting beta, tau, loss weights, or calibration sample count.
