# Selected-dimension numerical validation

Protocol: coco80-strict-tsfl-selected-dimension-v1. Evidence: COCO_SAFEBG_DIMENSION_CPU_NUMERICS_20261025.json and COCO_SAFEBG_DIMENSION_REGRESSION_20261025.json. All values below are measured CPU implementation checks, not COCO detection results.

| Check | Actual result |
|---|---|
| Independent frozen original SafeBG reference | commit3a8db4c, source SHA9c6dca9e… |
| VOC24 float32 and float64 outputs | max absolute difference0 |
| VOC24 all returned tensor/class diagnostics | exact equality |
| VOC24 eager cache projector bytes/hash | exact equality |
| COCO39 direct versus bounded cached projection | exact output/diagnostics |
| COCO39 selected projector | rank8, Gram error <=1e-5 |
| COCO39 norm restoration | within original float32 roundoff1e-6 |
| Non-target conflict inputs | all79, class79 dominant probe ranks first |
| Synthetic actual Fourier primitive → L_BG gradient | norm.021624747663736343, finite/nonzero |
| Synthetic actual Fourier primitive → L_rel gradient | norm.022693516686558723, finite/nonzero |
| P3 independent coordinate/loss/conflict check |64D/48 selected, exact before/after |
| Invalid dimension/basis/rank cases |6 rejected, no silent fallback |
| Bounded fixture cache |3entries,3984bytes, exact bound |
| Related regressions |55tests,0failures,0errors,0skips |

The gradient probe uses real unchanged Fourier/kproto-loss primitives with synthetic selected features. The learned-Teacher full native loss chain and current-candidate CUDA smoke/review are OPEN. Do not infer formal readiness from these component results. The regression runner retains all generated fixtures and prior failed import logs; no deletion/cleanup took place.
