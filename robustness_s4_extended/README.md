# Current recovery03 — original S4 selected

Full completed and was independently accepted/published. Frozen selection chose original S4. Selectedv2 failed its first Cutout smoke on native float32 boundary roundoff; new runtime40d2/confige35f fixes geometry-copy validation and retains the exact immutable Decision SHA630c8.56local+56server CPU tests pass; current three GPU smoke groups and one new extended review are required before paired formal training. OP remains blocked. See ../../EXTENDED_CONTINUE.md and reports/CUTOUT_BOUNDARY_REPAIR.md. Prior preparation notes follow.

# S4 extended defenses — independent successor

Latest authorization (2026-10-22): automatic S4/Full selection supersedes the
original-S4-only preparation below. Read reports/AUTOMATIC_METHOD_SELECTION_PROTOCOL.md.
The selected batch follows Full victim_recovery03, independently accepts its
complete generation plus200-epoch victim, then freezes one S4_FULL_DECISION.json.
Full wins only if S improves by more than1e-9 with PersonAP50<=S4,
NTmAP50>=S4, WCD<=S4 and identical protocols. Invalid Full blocks all new GPU work.
All defenses read the same Decision; no method is selected yet. OP remains disabled.
The previous no-job waiter is retained and normally superseded by the versioned
selected-method successor; no prior training is rerun or modified.

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

Actual deployment (2026-10-22): server successor `s4_robustness_extended_seed0_20261022_v1` is healthy / WAITING_PREVIOUS_PIPELINE, controller2875690/start326511739, active_job=None/jobs{}/zero new GPU work. Runtime source `37e0ae7a35baa32b7096239c07a30c76bdaa88b6`; config `a6a9ee615af7baf4dfbc149efe39545ac62c2892b3125906c8a02c29876506b1`. Both local and actual server31CPU contracts and ordinary independent-scope/singleton/completed-job-adoption passed. See reports/DEPLOYMENT_STATUS.json and CPU_SERVER_SMOKE_VERIFIED.json. Later documentation commits do not change this deployed runtime.
