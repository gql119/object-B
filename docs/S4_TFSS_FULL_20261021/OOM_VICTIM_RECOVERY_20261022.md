# Full Victim recovery after external OOM termination

Authorization: the human's 2026-10-22 instructions explicitly request diagnosis,
repair, continued experiments, and autonomous handling of later problems.

## Evidence and cause

The original bound workload PID2829581/start325195800 and executor2829570 are
absent. No Victim epoch row or checkpoint was produced. Its console ends at
validation RAM caching at 2026-10-22 17:51:10 HKT, without a Python traceback.
The same user's readable systemd journal records at 17:51:42 HKT:
`tmux-spawn-d98c5d70-b330-4571-8db6-864baef0344e.scope:
systemd-oomd killed 18 process(es) in this unit`, result `oom-kill`.
The original control process shared that failure domain, so its persisted
`running` status could not be updated after termination. The earlier failed
system journal query reflected journal visibility, not absence of an OOM event.
No sudo, environment change, resource-limit or OOM protection override is used.

The noise generation completed all24epochs/9144updates. A CPU-only read checked
the final training checkpoint's complete Adam/scheduler/RNG state and finite
Fourier/optimizer tensors. Training-checkpoint SHA:
79efe7cd0465cb7682d331194aeed2cc9af44a7adb50fee5c709855838ca0bf4.
Export parameters SHA:
90e9481fd093c6d43267b0f881c27cf4ea9e07e8108fd3b84b144e7d829f03fc.
Full16551train/4952cleanval/13256PersonGT pixel and label audits passed.
Manifest SHA77c092982cd3e259ea78424768af2b90d880d5f3bcbe215e71b8aa7de3f4c25b.

## Recovery contract

- Retain the failed original run and every completed generation/data artifact.
- New Victim RunID: s4_tfss_full_overlap_seed0_20261021_victim_recovery01.
- Rehash the completed Full images/labels and unchanged clean validation before
  use; link them into the new output without copying or regenerating images.
- Run the exact reviewed `train_evaluate` and `evaluate_completed` functions
  from scientific source1d10ad620ef01aad20e8c85d49465473d72c9a4e.
- Fresh random-YAML seed0 initialization and full200epochs. This is explicitly
  **not** an exact batch/checkpoint continuation of the interrupted Victim.
- Frozen protocol configSHA2bedd986d44dce40d3815093da76ef3d259aec83fab7f0c6518d0f7795b6675c
  is unchanged, including batch36/workers16/cacheTrue/AMP/nativeSGD/LR/augmentation.
- The original24epoch generation is never called by the recovery wrapper.
- Use the existing production-tested ordinary systemd job scope and independent
  supervisor. A child OOM therefore leaves an exit/scope receipt while the small
  controller can report failure. This does not prevent future external memory
  pressure and does not weaken OOM protections or guarantee uninterrupted training.
- Preserve existing capacity/I-O admission; CPU/GPU utilization remains telemetry.
- Bind generated-data scientific identity and new lifecycle runtime separately.
  The same frozen scientific evaluation/report functions consume only last@200.
- Extended defenses must follow the new completed recovery output and independently
  verify its generation lineage and full200 evidence before any new GPU work.

## Targeted validation

Six CPU lifecycle contracts pass: frozen training/safety source byte identity,
same-output/outside-root rejection, live-parent refusal, artifact-drift refusal,
and no noise optimization/export/resume entry. Actual source-bound ordinary-scope
CPU smoke is required before recovery launch. No second Full scientific review:
numerical source and parameters are unchanged; this is lifecycle/binding repair.

An initial local test harness attempted its default temporary directory, which
was inaccessible under the sandbox. Its failed cleanup also hit permission
errors. The harness now retains each case under execution/recovery_cpu_cases and
performs no cleanup. This incident preceded any recovery deployment or launch.

## Recovery01 retained; detached-session correction

At21:27:24HKT, the recovery01 tmux scope was itself oomd-killed (two processes).
The separate worker scope then ended without an exit receipt. The first wrapper
called the supervisor synchronously, unlike the original production Pipeline's
detached-supervisor path. Scope isolation alone was insufficient for the control
session failure. No recovery01 epoch/checkpoint or official metric exists.

Recovery02 adds a separately scoped controller behind a tiny RRCTL bootstrap,
ordinary scoped supervisors with start_new_session=True, and scoped CPU audit/
report jobs. Large dataset hashing no longer charges the control tmux scope.
A synthetic controller exits itself with code24 while the exact detached
supervisor/child completes and records exit0. This stronger actual-launcher
CPU smoke must pass before the unique recovery02 launch. No real OOM injection,
kernel/user memory policy modification, other-user action, or hyperparameter
change is used. Recovery01 and its failure logs are retained.

