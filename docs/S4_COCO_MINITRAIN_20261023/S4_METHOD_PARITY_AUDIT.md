# Method parity audit — risk extension candidate

Preserved: original8/4/0,Qt/W/eta.85,TSFL39/VOC24 selection rule,SafeBG rank8/alpha.5/top3/K8/norm restoration,P3 coordinates,all original losses/weights,Frozen/Adaptive/Fourier/EOT/Adam/schedule,full25K/COCO80/originalval5K. No native Ultralytics TAL changes or framework-method source changes.

Declared difference: for a valid GT lacking native and independent native supervision, the NT clean-input sensitivity uses an explicitly GT-conditioned independent detection-risk loss. Its hard GT classification targets/uniform candidate weights are new in that previously undefined branch. Therefore global equality to the original strict estimator is NOT claimed.

Normal covered calls delegate unchanged; actual output/gradient equivalence remains to be measured. Instance-local cache dependencies isolate this branch without changing originalQ/map/cache code objects or module globals. Cache identity includes new estimator/adapter/spec SHA. Full preventive source/consumer checks and actual production sinks remain pending.
