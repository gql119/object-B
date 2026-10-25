# Original S4 — COCO80 Dimension-Compatible Implementation: batch16 numerical repair

COCO uses39 retained TSFL channels and VOC uses24, following the same Strict TSFL rule. This is not Full/S5/CUP. The same ordered rank8 SVD, alpha.5, active3, norm restoration, original losses, P3 Orth,8/4/0 and eta.85 are preserved. Only internal SVD arithmetic for nonlegacy selected dimensions uses float64 followed by the original output dtype. The24D arithmetic/cache hash remains exact; no numerical threshold is relaxed.

| Actual evidence | CPU16 | CUDA16 |
|---|---:|---:|
| Native EOT batches |16,16|16,16|
| Native model forwards |12|12|
| Actual Fourier Adam updates |1|1|
| L_BG to64x3 coefficient gradient norm |.0236549321562|.0234878156334|
| L_rel to64x3 coefficient gradient norm |.0235552098602|.0233481898904|
| Original failing ordered38,24,26 Gram |5.96046447754e-8|1.19209289551e-7|
| Maximum native cached Gram |2.98023223877e-7|2.38418579102e-7|
| Native cached projector ranks |all exactly8|all exactly8|
| Frozen Teacher/checkpoint unchanged |yes|yes|

Both probes preserve the complete13630-image eligibility list and original NumPy seed0 permutation; a passive native image-loader trace independently matches all16 original first-batch IDs. Both retained current-source release manifests are byte-identical and bound to dcd91b33d90dfd2472b8eacb843a994a4801d1cf, release63eb86142add014c647558d256873c94e2d480023780903761b45423ed5ac80f and phaseconfig9f2830c5d6a2a608f881cb5d7c2616bc77f6340ca597ae74ab4070229361ce4a. Every actual forward tensor is on the requested device; CUDA index0 is measured. No official metrics/ingest, val images, Teacher updates, data transfer or cleanup occurred.

Main-agent single-thread CPU read-only audits verify138 actual source and7 native hashes, checkpoint integrity, saved coefficient-gradient arrays and mean norms, original first-batch order, native optimizer-step/delta diagnostics, Python/NumPy/Torch RNG (plus CUDA RNG in GPU run), and immutable failed-v1 evidence. Neither audit initialized GPU. Complete small reports were pulled once; checkpoints/images remain on server. Initial local audit plumbing failures (Windows command length; treating an inherited float32 display mean7.999999523 as an integer rank) were corrected only in the verifier. Rank admission remains exact per native cached integer projector, with no threshold or CSV alteration.

Exactly one independent review remains on record. Its original e891024 negative F01 verdict is retained under DIMENSION_RECOVERY01_20261025; main-agent repair closure links actual direct coefficient gradients and categorical per-EOT sidecars, plus the later numerical incident. No second/closing reviewer was called. Failed originalv3 and dimension-v1 remain unchanged and dead. The unique new v2 controller reuses accepted Teacher/eval/reconstructed calibration and receives a fresh bound READY through the original serial mechanism. It must be inspected after launch rather than restarted.

The experiment is not finished. Official S4 victim AP/PSNR/LPIPS results do not exist yet. Fourier24, fresh poisoned25K export, matched freshClean200/S4200, full originalval5K officialCOCO80 evaluation and final report remain governed by the same sole monitor and their actual phase gates. No single-seed significance/causal claim or AP tuning is made.


Actual formal first-step proof: the v2 controller automatically consumed READY and entered RUNNING/fourier24. Controller63444/start357606197/uid1014 and child64595/start357618769/uid1014 were independently alive;27 complete native CSV rows show27 actual optimizer updates in epoch1 (latest step26), finite L_total1.56020736694/gradient.05552721024/outside0. This proves recovery past the old first-batch failure; it is not completed24-epoch generation or a victim/AP result. Runtime source remainsdcd91b3, controllerconfig8228eca879de4619a01080f8d3fb959df555a4d9e0d3b2d764dbe83f7c82e3c8.
