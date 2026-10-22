# S4-TFSS-Full final accepted results

Full completed and passed independent main-agent acceptance after a fresh recovery03 victim. The one experimental change was Person–NT overlap support0.5→1.0, with original TFSS W=1−0.15Rnt(1−Qt), eta0.85, STE2.01/255 floor, all losses and native training settings preserved. The generated Full data are independent outputs; original S4 inputs remain immutable.

| last.pt@200 metric (fraction) | Original S4 | Full | Full minus S4 |
|---|---:|---:|---:|
| Person AP50 | 0.0281588954 | 0.0731859539 | +0.0450270585 |
| NT19 mAP50 | 0.6937870442 | 0.7033112636 | +0.0095242194 |
| NT19 mAP75 | 0.5152513353 | 0.5298847200 | +0.0146333847 |
| NT19 mAP50-95 | 0.480527673 | 0.492564632 | +0.012036959 |
| All20 mAP50 | 0.660505637 | 0.671804998 | +0.011299361 |
| P | 0.9675567540 | 0.9156788690 | −0.0518778850 |
| U | 0.9051813786 | 0.9176075922 | +0.0124262136 |
| S | 0.9353302965 | 0.9166422161 | −0.0186880805 |
| WCD | 0.2959510971 | 0.2528753477 | −0.0430757494 |
| TCD3 | 0.2644817012 | 0.2387982511 | −0.0256834502 |

P/U/S use the same hash-locked original Clean denominator and signed fractions without clipping; undefined denominators remain null. Full improves average non-target retention and worst relative collateral damage, while Person AP50 increases and protection weakens. It therefore does not meet the frozen conservative advantage rule. Selected Method is **S4_ORIGINAL**. This is the pre-registered scheduling choice, not a claim of universal method superiority. The immutable original Decision SHA is630c8ad65d110c115a72dc7ea50d88c55a2a1ffaac79172d239b0d02a0e612c1; rule SHAe9b2dcbe5ebe0aa7dd1b19150b74be1d02eccbb8b6e9a350580d735504bfe418. The repaired extended run inherits its exact bytes; it does not reselect.

Source and lineage: scientific1d10ad620ef01aad20e8c85d49465473d72c9a4e/config2bedd986d44dce40d3815093da76ef3d259aec83fab7f0c6518d0f7795b6675c; recovery lifecycle dc60396d044cefe08fd03ae6b126d33135ff11b2. Original Full generation v1 completed24epochs/9144Adam updates; noise checkpoint79efe7cd..., export global parameters90e9481f..., manifest77c09298..., integrityf56b4efd.... The full SHAs and cache/teacher identities are in FINAL_ACCEPTANCE.json and provenance. Training uses fresh random YAML YOLOv8n, seed0/200/batch36/workers16/AMP/nativeSGD/cacheTrue/640 and exact original LR/scheduler/warmup/base augmentations. Validation is the untouched4952clean images; last@200 is primary, best is separately archived secondary. Final native CSV and embedded last history both contain200 epochs. Last SHA2a831ea516b4c9d752465ea0c5fab628502788fd6a1e6caf34af931db489c6c6; best SHA5e5fd09cc68e3befeb8f69d75b612f496fe3a633ab3b05203b5c60c6a803dbe5. The full epoch200 state retains optimizer/scheduler/scaler/Python/NumPy/Torch/CUDA RNG and finite tensors. Latest observer completed_epoch201 is a callback counter after native finalization; actual200-row CSV, embedded last and epoch200 full-state define completion.

Data audit:16551training images,4952clean validation,6095Person images and13256Person GT. All13256target GT changed; decoded PNG values obey8/255 within target union and zero outside; labels and validation bytes unchanged. Geometry-derived overlap exists in3835images, absent in2260Person images. Mean overlap-region RMS across overlapping images rises from0.01500136(original S4) to0.02923768(Full), with area-pooled RMS0.01510432→0.02953166. Full overlap mean W_U=0.95904427; observed range includes0.85–1.0. Average per-image PSNR across target images is37.5384dB vs original38.9250dB. These are geometric/pixel diagnostics, not causal mediation evidence. All19NT classes, precision/recall and best-secondary arrays are retained in the JSON/CSV artifacts, including unfavorable classes.

Recorded victim training compute: final AMP observer elapsed14807.17s (4.113h),92000batches,46423native updates,21legal overflow skips, final consecutive skips0 and finite model/EMA/optimizer. This measurement excludes generation, export, resource waits, evaluation and failed attempts. No total end-to-end cost or statistical replication is implied. Earlier original victim/recovery01 external oomd/session failures and recovery02 bootstrap guard failure are retained; recovery03 retrained a fresh victim from epoch0 on the immutable completed Full dataset, without claiming exact prefetch-RNG resume.

Scientific interpretation: Full does not empirically replace original overlap protection under this single locked protocol. Its weaker target suppression alongside better NT retention is a measured tradeoff. The experiment alone cannot identify why it occurs; AP does not establish background absorption, selective causal decoupling or overlap-mediated effects. CPU random-teacher contracts were implementation tests only; formal GPU generation used the hash-bound pretrained teacher. SSIM was not run because the validated tool was unavailable. LPIPS is unavailable without a validated existing implementation. No extra seed, tuning, AT, additional method or rerun was used to improve these results.

Next: the independent extended server recovery owns selected-source validation, three bounded GPU smoke groups, exactly one extended scientific implementation review, then Cutout→CutMix→MixUp matched Clean/selectedS4 pairs. OP remains BLOCKED_PROTOCOL_APPROVAL. Full is complete and must never be launched again. This archive contains only code-linked configuration, small results and receipts; no data, weights or profiles.
