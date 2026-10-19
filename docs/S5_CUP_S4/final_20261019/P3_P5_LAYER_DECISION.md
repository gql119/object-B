# Stage C: retain P3 for S5

Decision: **P3/model.15 full-channel Frozen L_ret**. S5 is S4 + CUP-S4 class aggregation, without a retention-layer change. No additional formal S4-P5 control is selected or needed. This is the preregistered mixed/weak-evidence choice, not a claim that P3 is universally optimal.

Evidence: RunID s5_p3_p5_train32_layerprobe_seed0_20261018_v2; sourcebb5cd1e56e5332c34bf41463525e429d08bfc310; completed exit0, eight-artifact contract verified by RRCTL completion health/wait/pull. Actual S4/P3-noop loss, Fourier gradient, delta, W, optimizer/scheduler/RNG/frequency state max difference0. Six independent trials each5 Adam updates; 72 EOT observations including no-update final observations. Both layers preserve all original valid P3 ROIs in this TRAIN32 (missing0); this does not establish universal small-object coverage.

| Final post-5step paired measure | Batch0 P3 | Batch0 P5 | Batch1 P3 | Batch1 P5 |
|---|---:|---:|---:|---:|
| Protection objective proxy | .9907076061 | .9907309711 | 1.0262875557 | 1.0261749625 |
| Same-Frozen NT native GT signed loss delta | .0085476041 | .0081641177 | -.0012727261 | -.0009169161 |
| lambda_ret gradient / protection norm | .00566564 | .04577500 | .01034654 | .09145324 |
| Opposing protection projection ratio | .00177213 | .02117830 | .00662198 | .05525145 |
| Retention/protection cosine | -.31214356 | -.46545871 | -.63953365 | -.60283991 |

P5 improves the final average native-GT delta in batch0 but worsens it in batch1. Batch0 bicycle is slightly worse; batch1 aeroplane/bicycle/bus/horse/pottedplant worsen while boat/car/dog/train/tv improve. Worst-class native-GT delta improves slightly in both batches (batch0 sofa .03368259→.03252077; batch1 train .01839195→.01769788), but this does not outweigh mixed mean/class/size effects and larger protection-opposing gradient magnitude. Small-object same-layer drift is not consistently improved. Full unfavorable values remain in layer_probe_20261019 CSVs.

Static first-step comparisons show P5 can have a more favorable cosine at the initial state; this does not persist consistently as an advantage after5 updates. Raw P5/P3 drift differs in dimension and response scale; neither raw nor per-channel MSE is a detector-utility ranking.

Runtime: actual model.15 C2f [16,64,80,80], model.21 C2f [16,256,20,20] for640 input. Measured P3 observer trials54.49/63.97s and P5 observer trials50.67/56.52s; peak allocated P3≈9.41GiB and P5≈9.41GiB. Timings include observer autograd and cache overhead, not production throughput. Cost similarity does not establish scientific benefit.

Limit: two fixed train batches, not independent statistical replications; EOT/ROI rows are dependent. The original S4 loop over a fixed16-image development subset advances its development epoch each pass: after5 updates epoch4/cursor16/scheduler4, lr.000933682574873297, Adam5 for all6 trials. This is a matched bounded development experiment, not the first5 updates of the full6095-support-image formal schedule and not convergence/AP evidence. Formal S5 retains the full registered support order and24epoch/9144 schedule. Checkpoint metadata/RNG/Fourier/Adam finite and exact boundaries are retained.

Next: implement only selected P3 class weighting with detached feature-drift history; validate current-source S4/off equivalence, class priority, effective gradients, resume, budgets and native AMP, then one independent S5 scientific review before formal execution. CUP/StageE/formal S5 have not run.
