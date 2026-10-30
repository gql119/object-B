# Selected OriginalBatch36 fallback scientific review

- Review mode: scientific_review
- Reviewer: /root/at_fresh_fallback_scientific_review_20261030, separate read-only Codex/GPT agent, inherited model configuration.
- Result: **scientifically_correct**
- Decision: **allow_run**
- Code snapshot: `87456fa7a57206551b0e3be0253cc4866f0cdcf1`
- Diff base: `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`
- Raw packet SHA256: `cb630f7ea0854708e8d8572f35f28afb7f9101f204ec36285798358d8ebc296c`
- Approved basis: `USER_AT_TASK_20261027.md` and `USER_DIRECT_TERMINATION_AND_RESTART_20261030.md`.
- Blockers: none.
- Validation gaps that prevent starting the reviewed scientific run: none.

The authorized fresh fallback retains OriginalBatch36, PGD10, complete GT and four independent seed0 random-YAML 200epoch arms. All19 scientific package/protocol hashes match current-source CUDA evidence; committed scientific directories and model YAML are unchanged from the accepted base. The exact command starts each formal worker without fixture or resume input. Candidate02/03, new batch sizes, ST fine-tuning and abandoned checkpoints are not adopted.

Intent alignment, canonical code, critical data flow, computation sink effect, runtime scientific state, measured baseline reuse and experiment identity are **correct**. The detailed verdict and evidence mapping are recorded verbatim in the corresponding JSON.

The current real CUDA smoke records26 native outer batches across four arms,260 PGD forwards, one actual SGD update per arm and22 legal AMP skips, with finite losses, batch36/imgsz640/workers16 and no validator forwards. All four initial parameter hashes match. Full transformed GT reaches native box/cls/DFL and TAL; inner PGD isolates parameter gradients and BN statistics and enforces strict4/255 or8/255 budgets after native augmentation/normalization. Native outer AMP/SGD/EMA computation is retained.

Baseline reuse is established by the measured probe, not inferred from code: a separately instantiated pinned DetectionModel on the same frozen initial state and actual first PGD batch gives loss453.893798828125 on both paths, difference0 at tolerance0. This is one bounded pre-update native-loss comparison, with no AP or efficacy implication.

Historical **AT-SCI-001 remains a preserved negative finding** in `execution/AT_SCIENTIFIC_REVIEW_20261029.json`: the former completed-epoch resume lost optimizer-attempt phase. Current accepted code and the fresh current-source CUDA phase proof close that defect. At post-warmup accumulation2, stored phase56 produces subsequent phases[56,58] on both uninterrupted and resumed paths: first batch accumulates, second performs one SGD update. Model, optimizer, scaler, scheduler, EMA, gradients and PGD RNG match exactly. workers16 prefetch bitwise trajectory equivalence remains explicitly unverified.

Formal evaluation uses clean4952 VOC images through a plain validator, last.pt@200 primary and best secondary. Full20class AP extraction,19non-target arithmetic, matched-budget CleanAT denominators, signed unclipped paired scores and null invalid denominators are correct. ST references remain separate. No official AP or completed200epoch result is claimed.

Local retained validations report passing source/CPU reuse and the two-site diagnostic metadata binding. Reviewer checks independently verified the committed diff, all19 current scientific hashes, raw packet hash, retained full-GT evidence hash and actual baseline invocation. No new tests, models or GPU work were launched. Current smoke reuses retained full16551/4952 label/split/acceptance evidence and hashes720 fixtures per arm; decoded static budget checks are samples, not a fresh full pixel census.

The readiness checker hashes canonical JSON; this verdict intentionally hashes raw packet bytes. Their different digest representations are not a packet mutation. RRCTL, process control, locks, transport, monitoring and schedulers were excluded, and no additional control review was performed.

Exact gated official command:

```text
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -B execution/direct_stop_20261030/fresh_serial_queue.py --out /project/guoqinglong/projects/s4_at_robustness_20261027/outputs/s4_at_serial_seed0_20261030_fresh_v1/result --run-id s4_at_serial_seed0_20261030_fresh_v1 --source-commit 87456fa7a57206551b0e3be0253cc4866f0cdcf1 --preflight /project/guoqinglong/projects/s4_at_robustness_20261027/preflight_20261030_fresh_v1/accepted.json
```

The scientific verdict permits this reviewed command through its actual gates. It does not itself launch training or assert a speedup, defense effect or completed experiment.
