# Compatibility audit

Parent is actual server original S4 generationb1e2951, not Full/S5. Seven schema/diagnostic files change only dataset schema, diagnostics, cache selection, and dataset-derived schedule metadata. New branch only; original worktrees remain byte untouched.

TFSS clean maps/NT gradients and decoded budget audit accept target_class_id/num_classes with original14/20 defaults; COCO explicitly0/80. Native all-GT TAL, same-TAL uncovered-GT handling, per-class BCE excluding target, box/DFL gains, per-object/class averaging and original normalization remain. Actual CPU random-YAML80-class loss/input gradient runs on a96x160 image, returning only class79 sensitivity in original coordinates. For VOC20, maps and matching records equal the old Git source exactly on the same actual native model/input.

Both semantic calibration builders now use configured class count/names. No VOC channel IDs are copied. Strict gates and fractions remain original; P5 remains fixed. Existing rank8/24D guard remains: if newly calibrated strict COCO features are not24D or any class has insufficient valid/rank evidence, block. Never relax TSFL or omit an NT class. P3 orth/ret and class conflict ranking stay unchanged.

Safe-BG eager cache retains969triplets*6=5814 ordered projectors for19classes.79classes would require79079*6=474474,81.61x as many. q tensor alone is24*8*4 bytes:~4.26MiB versus~347.52MiB, plus singular values, Python objects, SVD workspaces and original all_q duplicate. New on-demand cache keys the EXACT ordered active triple, uses the same torch.linalg.svd/concatenation/rank threshold and projector, and stores at most8192entries (~6MiB q tensors; no preallocation of474474SVDs). Repeated calls after eviction recompute identical ordered math. Basis hashes are independent of insertion/call order. CPU tests compare original eager19 variants and complete directions/diagnostics with exact equality. CUDA equality remains PENDING.

Optimization metadata uses configured class names/target. VOC's original24/9144 schedule gate stays unchanged; COCO keeps24epochs/batch16/EOT2 but derives step count from the clean COCO eligible-image manifest, not VOC6095 images. No old Fourier parameters or teacher/coordinate/cache output is reused.

Unchanged original code includes support/floor/JND/Fourier composition, spatial formula, Frozen/Adaptive inner update, all scientific losses, projection formula, soft05 removal and SafeBG class ranking. Full model/loss-chain CUDA integration, isolated dependency/source binding, complete server data conversion/export and final orchestration still require completion before formal runs. CPU contracts are not a claim that those pending paths have passed.


Original Victim component: ue_framework/s4_victim.py and s4_amp_observer.py are copied exactly from server ab626917, not Full/S5. This separate original-S4 AMP repair replaces the older generation-time lifecycle wrapper; scientific generation remains b1e2951. Current production pipeline integration remains pending and cannot launch from legacy VOC count-locked entrypoints.


Original Victim component: ue_framework/s4_victim.py and s4_amp_observer.py are copied exactly from server ab626917, not Full/S5. This separate original-S4 AMP repair replaces the older generation-time lifecycle wrapper; scientific generation remains b1e2951. Current production pipeline integration remains pending and cannot launch from legacy VOC count-locked entrypoints.
