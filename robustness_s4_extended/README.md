# S4 extended defenses — independent successor

Start with `reports/EXTENDED_ROBUSTNESS_AUDIT.md`, `reports/REFERENCE_IMPLEMENTATION_AUDIT.md`, `reports/OP_DETECTION_ADAPTATION_REVIEW.md` and `reports/EXTENDED_ROBUSTNESS_TEST_REPORT.md`.

Only original S4 Safe-Soft05+TFSS-Utility training data feeds Poison. S4-TFSS-Full is the queue predecessor, never the Poison source. No existing process, environment, original image, label, noise or checkpoint is modified. All four methods are separate paired Clean/S4 experiments, seed0/200/random-YAML/native AMP/batch36/workers16/640/last@200, clean4952 validation.

CPU-only preparation:

```powershell
python -B -m robustness_s4_extended.validate_cpu
python -B -m robustness_s4_extended.visualize_cpu
```

The server successor is `python -B -m robustness_s4_extended.successor --expected-source <committed SHA>`. **Do not run this manually for an already bound RunID.** Deployment and read-only status use the retained RRCTL control wrapper `execution/extended_control.py`. Never run a second successor or direct training command.

Queue: Full native complete → independent full acceptance adapter with actual data/checkpoint/200CSV/hash verification → current-code/source/data CPU validation → three isolated GPU smokes → **one current-source scientific review receipt** → Cutout Clean then S4 → accept → CutMix Clean then S4 → accept → MixUp Clean then S4 → accept → archive partial report and `BLOCKED_PROTOCOL_APPROVAL` for OP.

GPU smokes and all new training remain deferred until Full acceptance. The project pre-run skill requires GPU evidence followed by one independent scientific review; the server waits persistently for that receipt and cannot fabricate it. The existing sole Codex monitor may supply the review closure once GPU evidence exists; no continuously connected local machine/VPN is needed to keep server waiting/training alive. Training/evaluation/stage transitions themselves run on the server without manual stage launches. If review service is unavailable, it remains `PENDING_SCIENTIFIC_REVIEW` rather than pretending all gates passed.

Native `close_mosaic=100` is retained: CutMix/MixUp are disabled for the last100 epochs; Cutout runs all200. Existing Mosaic/HSV/flip/affine probabilities are preserved. A single new defense is active in each arm. Matching Clean controls are retrained; none of the historical eight-defense Clean denominators is reused.

OP has no production worker route, model-training CLI or enabled default. Approving an arbitrary Boolean cannot run an unspecified detection protocol. A selected/approved protocol must be implemented, validated and source-bound before any OP GPU/model/full-dataset work.

Research conclusions require complete paired formal acceptance. CPU tests and visuals are correctness evidence only, with no AP, robustness-effect, or causal-mechanism claims.
