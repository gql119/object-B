# S1 correctness audit

Stage0 source/config/checkpoint/metrics provenance is located. Historical-source vs restored-source 5-step GPU equivalence passed in `s1_parent_equivalence_train16_seed0_20261014_v2`: max_loss_difference=2.980232238769531e-8, max_tensor_difference=3.725290298461914e-9, tolerance=1e-6. Evidence binds actual source imported in separate processes, checkpoint/prototype/config and same 16 train ids. Tensor comparison covers target/utility gradients, total gradient, poison and Fourier updates for every step/EOT. v1 path-binding failure retained; v2 exited0/rrctl completed.

CSB masked normalization, no-NT fallback/error distinction and PNG per-pixel quantized bounds: 3 local tests passed. CSB spatial GPU audit, projection equivalence, exact resume, 5-step paired diagnosis and independent review are not yet complete. No formal S1 or victim is running.

Spatial32v1 GPU audit passed and frozen teacher state unchanged; 26NT images43GT, all10positives. Modelpeak279684608bytes, mean sensitivity compute .1027345s/image. Eight geometry categories covered. No degeneracy, invalidGT/gradient or zeroPerson budget. No validation images selected.

Mechanism32v2 passed after shared-poison binding fix: native-loop vs fixed source loss/coeff maxdifference0, CSB3→5 actual checkpoint replay allfields0. Pixel export checks8images/arm passed; no fullgeneration/victim. Source commit143b4ed. New end-to-end pipeline, addedNT confidence/localization diagnostics, heatmaps, locked metadata and production-frequency benchmark are compiled/locally tested but pending current-commit GPU probe and independent review. No allow_run claimed.

Current local validation:5 targeted tests passed, covering exact45parent-source bytes, unchanged locked scientific fields, spatial fallback/bounds/PNGquantization and orderedAP75 column/finite validator outputs. All new computing scripts compile. Full-cache6095 and full materialization/victim are not yet run.
