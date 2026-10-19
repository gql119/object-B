# CUP-S4 adaptation and code diff

Approved method adaptation: original S3 post-CNLP-learning query damage is replaced by selected P3 stopgrad feature-drift priority. No CNLP/virtual SGD model is imported. The original P3 GT loss and all S4 scientific source bytes stay intact; only its class aggregation is weighted.

New actual sinks:
- ue_framework/methods/s5_retention_core.py: original Frozen _batch_loss plus2 observation sites and1retention reduction sink; parent AST regression passes.
- s5_retention_probe.py: GT/class/layer observations and original instance-baseline reconstruction; actual S4/P3 StageB scalar/gradient/delta/state maximum0.
- s5_cup_history.py: detached raw/normalized EMA, fixed per-class scales/fallback, relative pi-weighted projection, identity-checked serialization.
- s5_cup_trainer.py: return exact original loss when disabled/neutral; actual pi*w class residual; classwise and effective gradient diagnostics; pending EOT state and Adam-boundary commit hooks.
- s5_optimization.py: inherited loop arithmetic; begin/end/empty-step hooks, CUP checkpoint and SHA-bound weight-journal replay. Exact line diff S5_OPTIMIZATION_PARENT.diff.
- s5_runtime.py: same exact fresh seed0 S4/off/CUP construction in smoke and formal paths.
- scripts/calibrate_s5_cup.py: fixed TRAIN32 rule, no forwards/updates/AP.
- scripts/s5_cup_probe.py, s5_cup_probe_checks.py: bounded actual threeway5step, no-op, nativeGT proxies, static norm-matched diagnostics,3to5replay and PNG/label/weight gates.
- scripts/s5_victim_smoke.py: original native AMP trial body with synthetic bound32→20 only; exact diff S5_NATIVE_AMP_TRIAL_PARENT.diff. Original S4 AMP/optimizer implementation untouched.
- scripts/run_s5_pipeline.py, s5_report.py: S5 identity-safe serial training/evaluation/reporting, missing historical optional aggregates stay unavailable rather than KeyError; no checkpoint/data overwrite or completed-stage rerun.

Params: all original S4 values unchanged; CUP-only beta.95, factors[.5,2], T1, normalizedclip55.70708166981073, scale floor1e-12 and measured12-class scales/7class first-observation fallback. These are disclosed and frozen before current-source GPU smoke. Detached priority derivatives arezero; original class Lret remains differentiable. Pi basis is instance counts, never silently1/K. P3 selection changes no retention layer and preserves the intended S4-to-S5 CUP-only attribution. Gradient magnitude is observed, not normalized away.

At publication/acceptance retain reviewed source SHA, source/config/calibration hashes and actual GPU receipts. Current implementation validation is CPU-only; no current CUP GPU/review/formal claim is made here.
