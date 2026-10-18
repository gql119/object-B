# Victim protocol equivalence

Local source fidelity: exact45 original S0 bytes pass; S4 method/map/config files are unchanged. Recovery fixed configuration carries the original victim_protocol object without modifying batch36, workers16, AMPtrue, seed0, randomYAML, SGD,200epochs, LR, cosine schedule, warmup/augmentation defaults, EMA, clipping or checkpoint cadence. The pre-existing S4 patience0 setting only enforces the requested full200epochs.

CPU numeric observer/native equivalence: exact model/SGD/scaler/scheduler states across finite updates; overflow skip and backoff preserve parameters, and the next finite update succeeds. No custom AMP optimizer exists.

Current installed environment + historical S0 checkpoint re-verification: NOT_RUN while RRCTL is unreachable. Historical contemporaneous Trainer source remains unknown; it will not be invented from a later archive.

Actual native/repaired CUDA loss/update/scheduler/EMA/scaler equivalence on identical transformed S0 batches: NOT_RUN. The smoke requires exact input and state hashes, not code-reading equivalence. Real S4 batches form a separate trial. Formal admission requires the exact current smoke commit, native Trainer source hash and full data audit receipt to match.
