# Safe-Soft05-KD-ObjBal formal result

RunID `safe_soft05_kdcls_objbal_seed0_20261008` completed on 2026-10-08
22:10:45 Hong Kong time. The isolated pipeline returned exit code 0 and each
stage (noise, materialization, dataset audit, victim, checkpoint evaluation,
comparison) returned exit code 0. This is one seed-0 detection adaptation;
the comparison is descriptive and does not establish a causal gain.

The new non-target classification KD uses the preregistered object-balanced
reduction and fixed 16-train-query gradient-calibrated coefficient
`lambda_KD=0.7160846249749698`. All other Safe-Soft05 settings, including the
original `L_ret`, were held fixed. The calibration gate, deterministic
parent-term equivalence probe, five-step production smoke, pixel gate, and one
independent `scientifically_correct / allow_run` review passed before formal
execution. In the 16-query calibration, every one of the 24 valid non-target
objects had ten TAL positives. On that fixed set, object-balanced and
positive-unit means coincide algebraically, so the gradient-tail change
cannot be attributed to the reduction.

Noise generation completed 24 epochs and 9144 steps in 28,404.33 seconds
(7 h 53 min 24 s), within the eight-hour cap. Materialization covered all
16,551 train images; train labels and stems were identical. The exhaustive
decoded-pixel audit passed the 8/4/0 uint8 target/overlap/outside limits,
and 13,256 person boxes contained changed pixels. The victim used the same
16,551-image train set, 4,952 clean validation images copied byte-for-byte,
random initialization, seed 0 and 200 epochs (the results CSV has epochs
1 through 200). `last.pt` is the primary evaluation; `best.pt` is secondary.

The primary matched clean-VOC20 validation comparison uses Safe-Soft05's
`last.pt` and KD-ObjBal's `last.pt`:

| Metric | Safe-Soft05 | KD-ObjBal | KD minus parent |
| --- | ---: | ---: | ---: |
| person AP50 | 0.028993 | 0.039584 | +0.010591 |
| person Recall | 0.000663 | 0.000692 | +0.000030 |
| non-target mAP50 | 0.691698 | 0.690701 | -0.000997 |
| all-class mAP50 | 0.658563 | 0.658145 | -0.000418 |
| P | 0.966595 | 0.954393 | -0.012202 |
| U | 0.902456 | 0.901155 | -0.001301 |
| S | 0.933425 | 0.927010 | -0.006415 |
| WCD | 0.281032 | 0.346698 | +0.065665 |
| TCD3 | 0.257851 | 0.271690 | +0.013839 |

The `best.pt` secondary evaluation gave person AP50 `0.041707`, person
Recall `0.000442`, non-target mAP50 `0.694325`, and all-class mAP50
`0.661694`. The target AP50 is higher than the matched parent under both
checkpoints, so this run does not show stronger target suppression. Mean
non-target AP50 changed little with `last.pt`, while individual classes
varied; dog AP50 fell by `0.052978` and horse by `0.025470` versus the parent.
The full 20-class AP50/Recall comparison is in
`experiments/kd_objbal_formal_20261008/voc20_per_class.csv`.

Source and evidence locks: config SHA256
`8107f4a3f7859b3d775cfa3c37d461a774bc379b2782206ec7b0c39b363fc741`,
calibration SHA256 `0ff854235663215da69b4fe04215c4a172885f165423b300f2864977903c9532`,
noise status SHA256 `2ee3d09743298d90d59caa3a27175cb0988efdcd820d0397d973c2334a2b4cbc`,
dataset audit SHA256 `a85851f000d4ba5e07d0ff5ee898787b3e122053ca97315dfb720fe30a7ffea3`.
The remote `last.pt` SHA256 is
`c7b4cf61385be01b6d739a54a6b0dcbdceca11193b6b9407721b2fc4adfb319e`
and `best.pt` SHA256 is
`b13c05101ed53de99defb2bb093cf4c67374ae1d5f0c5a8d73fb6756a9128bc2`;
the checkpoints and image datasets remain on the server and are not in Git.
The inherited `legacy_kproto_ret.py` is included with its exact remote-run
bytes (SHA256 `6bfd9155ea4ed4aa90dba655d835407c0bfad92737b153adc0060215c0020ccb`).
Its text is unchanged from the branch parent; only mixed original line endings
differ. A path-specific Git attribute prevents checkout conversion from
breaking the reviewed source hash.

Raw small JSON/CSV evidence is preserved under
`experiments/kd_objbal_prerun_20261008/` and
`experiments/kd_objbal_formal_20261008/`. LPIPS was unavailable and was not
substituted with another perceptual metric. A single seed cannot resolve
run-to-run variation, and neither target AP50 nor the gradient probe alone
proves background absorption or the mechanism of any class change.
