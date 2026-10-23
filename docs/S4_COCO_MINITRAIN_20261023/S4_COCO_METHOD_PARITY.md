# Original S4 method parity

| Component | VOC original | COCO adaptation / evidence |
|---|---|---|
| Scientific lineage | Safe-Soft05+TFSS-Utility, b1e2951 | same exact parent; excludes Full and S5 |
| Target / classes |14 /20 |0 /80 via explicit schema |
| Support | Person1, unionNT overlap.5, outside0 | same original _StrictTargetSupport; multiple NT overlap never.25 |
| Export |8/4/0 integer-byte caps, original STE2.01/255 floor | original composition and png_project unchanged; decoded synthetic tests |
| TFSS |1-.15Rnt(1-Qt),eta.85 | same implementation; current schema native input-gradient and default-VOC exact tests |
| TSFL | original strict P5 gates/fractions, actual24channels | recompute COCO; do not copy VOC indices; block incompatible strict result |
| Context |K8, cosine full-space match, BG/rel losses | original implementation untouched; rebuild clean25K prototypes |
| SafeBG | top3,rank8,alpha.5,norm restore | same formula/ordered SVD; bounded cache tested against original |
| L_orth/L_ret | P3,original top3/overlap weighting/weights | same functions; expand all79 NT bases/names |
| Adaptive | Frozen/Adaptive,one virtual step,alpha1e-4,original weights | exact unchanged code, new matched clean25K teacher |
| Fourier | shared original seed0/24epochs/Adam/batch16/EOT2/frequencies/JND/TV | same math; new initialization/cache/noise, dataset-derived step count |
| Teacher | VOC teacher | strict clean25K COCO80 random-YAML200 teacher, val=False; not public full-train2017 pretrained |
| Victims | random YAML,200,36,16,AMP/nativeSGD/640/scheduler/baseaug | same locked arguments, head80; fresh matched controls |
| Evaluation |cleanVOC4952,last@200 primary/best secondary | cleanCOCO val2017 5000, official COCOeval; both arms same protocol |

This table records intended parity and implemented component tests, not completed COCO research. Missing real teacher/full-data/actual CUDA loss-chain/PNG evidence must pass before production. No AP-based tuning, new loss/generator/seed, separate Adaptive support-query data, Full support or CUP is authorized.
