# S5 最终验收 — 2026-10-19

- 状态：COMPLETE / VERIFIED / ARCHIVED，远端 exit0，原绑定训练进程已退出。完整 24 epoch / 9,144 noise 更新、全量数据审计、随机初始化 YOLOv8n Victim 200 epoch、last 主评价与 best 次评价均完成。
- 唯一 RunID：s5_tfss_utility_cup_p3_seed0_20261019_v1；科学源码 e29d4662465159b5a1d1d7a5fd6c19c545f764e8；配置 SHA afdcd04e925d73f50eeb9d450bdbb259195d1500f61f1e46ecd08dd66807f0c8。
- S5 last Person AP50=0.02258628164286698，NT AP50=0.690105689344675，NT AP75=0.5102067823101584；WCD=0.31318165930240155（dog），TCD3=0.26835235162192933。
- 相比 S4：Person -0.5573pp，NT -0.3681pp，NT AP75 -0.5045pp，WCD/TCD3 变差。综合判据未通过；执行验收不等于研究成功。单种子，不作显著性/因果/跨种子结论。
- 37 个文件大小/SHA 核验、全类已有聚合和带符号指标本地复算通过；最终 checkpoint Adam/CUP9144、scheduler24、Fourier/RNG、200 行 CSV 与 last 内嵌记录核验通过。206 个 checkpoint 保留，无新优化或数据/权重传输。
- 完整报告：docs/S5_CUP_S4/final_20261019/S5_FINAL_REPORT_ZH.md；主代理验收 LOCAL_ACCEPTANCE.json；原始完整证据保留 BASE/s5_small_20261018/唯一RunID。
- S0–S4 继续保持已归档；不重跑、调参或自动追加实验。当前仅完成此前授权的小型结果发布，完成后关闭原唯一监控。

以下均为历史过程记录，以本节最终状态为准。

---

# Latest S5 progress verified 2026-10-19 16:17 Asia/Hong_Kong

Same sole boundPID1880295/startticks297568436 alive, victim_training. Victim CSV177/200 contiguousepochs, recordedlossesfinite; last/best currently provisional. Registered noise24/9144 and dataset integrity alreadypassed, auditSHA unchanged. Current stage evidence execution/S5_STAGE_20261019T081720Z.json. This is bounded stage/CSV/checkpoint-metadata inspection, not a fresh three-window resource audit or final checkpoint-content acceptance. Continue same process through200 and final evaluations/reports; neverduplicate/restart.

# Latest S5 stage verified 2026-10-19 14:22 Asia/Hong_Kong

Same sole RunID s5_tfss_utility_cup_p3_seed0_20261019_v1 and PID1880295/startticks297568436. Noise COMPLETE24/24epochs9144/9144updates,32807.92903734278s (registered schedule completion, not convergence). Full dataset audit passed16551train/4952cleanval/6095targetimages/13256PersonGTchanged; labels/cleanval/nonPerson byte-identical, outside0/per-pixel8-4-0. AuditSHA446158754c3d7c3540dc175432e4844c7dd434e182af94ac1a6dfa09888b3bd5. Victim RUNNING,81/200continuousCSVepochs,recordedlossesfinite; fresh randomYAMLseed0/batch36/workers16/AMPtrue/nativeSGD. Last/best files currently exist but provisional; no finalAPclaim. Latest read-only RRCTL evidence execution/S5_STAGE_20261019T062231Z.json and S5_MONITOR_20261019T062143Z.json. Actual3-windowCPU68.68/56.05/56.10%,IO.02/.04/.03%,ownGPUonly,activeguardsPASS. Source/configidentities exact. Pipeline's embedded resource block is launch-time metadata; use current monitor for current resources. No new launch/signals/remote mutation; final checkpoint payload/Adam/CUP/RNG acceptance remains pending. Never rerun noise/data or create another victim.

# Latest S5 read-only monitor 2026-10-18T19:22:45UTC

Same boundPID1880295/startticks297568436, sourcee29d466/configafdcd04e exact/remoteGit clean; healthy noise_generation epoch1,65/9144steps. Three samples progressed47->57->65; CPU100/47.61/100%, I/O0/.04/0%, no sustained contention. GPU onlyownprocess9471MiB; activeRAM/disk/source/config/PID safeguards passed. Retained immutable checkpoint metadata/hash recorded in execution/S5_MONITOR_20261018T192245Z.json (content acceptance not yet performed). No signals, launch or remote writes; unchanged healthy notification suppressed. Follow existing run, neverduplicate.

# Latest S5: FORMAL LAUNCHED EXACTLY ONCE

RunID s5_tfss_utility_cup_p3_seed0_20261019_v1, sourcee29d466, configSHAafdcd04e925d73f50eeb9d450bdbb259195d1500f61f1e46ecd08dd66807f0c8. WorkloadPID1880295/executor1880293; first-step RRCTL healthy. Current-source smokev2 passed92/98units, exactS4/noop0/PNG32/resume0/nativeAMPpassed. ExactlyONE independent S5 reviewer scientifically_correct/allow_run with no blockers; receiptSHA25663bc1f6500ca76bcc848dc4e9a1aae6730b49beff34ddca5afbaf5f4db4f1. Three-window resources and finalpreflights passed. Neverrelaunch. Follow existing binding through24/9144/full dataset audit/victim200/lastprimary/fullVOC20/reports acceptance/publication. ShortproxiesMIXED, noAPclaim. See execution/S5_CURRENT_STATE.json and S5_CONTINUE.md.

# Latest S5: v1 synthetic AMP harness failed; repaired candidate e29d466

40 noise updates/resume/PNG passed; synthetic observer asserted on accumulation-only batch11 before native optimizer attempt. Five targeted contracts pass; only smoke observer and smoke RunID metadata repaired. v2 fresh-source GPU pending; reviewer NOT_CALLED/formal NOT_RUN. See S5_AMP_ACCUMULATION_HARNESS_REPAIR.md.

# Latest S5: current-source StageE smoke RUNNING

RunID s5_cup_p3_train32_smoke_seed0_20261019_v1 launched once onef7889a; parent audit completed and actualS4reference batch0 active, RRCTLhealthy. WorkloadPID1852763/executor1852761. Source/config tested/committed bytes exact; scopedGitHubbranchremoteSHAverified. Budget<=98 (40noiseAdam,12postEOT,no-update,AMP5/5/16/max20) retainall. FormalS5/reviewer NOT_RUN/NOT_CALLED. Currentstate execution/S5_CURRENT_STATE.json; next actualsource gates then oneindependentreview and authorizedformal.

# Latest S5: StageD implemented; current-source StageE pending

P3 selected from completed realTRAIN32 StageB. New detached class-drift EMA/normalization/bounded pi weighting and CUP Adam-boundary/checkpoint/journal hooks are implemented; originalS4/45S0files unchanged. CPU14contracts pass; compile16/importexports71 pass. Selected-P3 config/calibration and scientific differences documented. Current CUP GPU/AMP/resume/PNG/independentreview NOT_RUN; formalS5 planned butUNLAUNCHED. Fresh resource audit underway; proceed to isolatedStageE using the actual formal trainer and native victim paths, then exactlyoneS5review and authorizedformal gates. User explicitly asks autonomous diagnosis/repair toward formal execution; do not ask repeated scope confirmation or relax correctness.

# Latest S5: StageC selected P3; StageD next

StageB v2 completedexit0,30updates/artifact8verified; actualS4/P3 maximumdifference0, all6modelhashpairsunchanged, finiteactualgradients and8/4/0 assertionspass. P5 nativeGTproxy mixed across batches with muchlarger opposing protection projection; retain originalP3 for single-variable CUP attribution. See docs/S5_CUP_S4/P3_P5_LAYER_DECISION.md. ActualCUP/StageE/review/formalS5 NOT_RUN. FormalS4/S3 untouched.

# Latest S5: Stage B v2 RUNNING

Unique `s5_p3_p5_train32_layerprobe_seed0_20261018_v2`, sourcebb5cd1e56e5332c34bf41463525e429d08bfc310, launched once; first-step RRCTL healthy, parent_setup, workloadPID1842376/executor1842361. Fresh resources and path absence passed. Scope is original TRAIN32, two independent16 batches, actualS4/P3-noop/P5 five updates each (30 total), CUPinactive, no formalnoise/victim/AP. v1 terminal import failure preserved, never relaunch. Track same binding, completion health->wait->smallpull; no layer decision yet.

# Latest S5: Stage B entrypoint repair

VPN/RRCTL restored; three-window resource audit passed 2026-10-18T17:42:46Z. Unique Stage B v1 launched once at17:43:25UTC and failed exit1 at parent_setup before model setup or optimizer updates: import of nonexistent `resource_guard`. Actual inherited production API is `resource_gate()`; only this import/call is corrected. All v1 evidence retained, never reuse its ID. Production dependency exports/imports check passed8; tests are running. Next commit fix and use unique v2 after resource/path checks. No layer choice, CUP or formal S5 yet.

# Latest S5: WAIT_FOR_VPN_ROUTE_RRCTL_NETWORK

Candidate Stage B code410212b38cb3349df4a5c7484596913818720707, local5tests/compile pass, RunSpec schema parse PASS. RRCTL resource query timed out before the first sample (exit255). Read-only local route inspection selects the WLAN default route for10.15.119.134 and reports only WLAN connected; no VPN-specific route observed. No S5 ready/launch, remote mutation or GPU work occurred. Resource availability is UNKNOWN. Preserve the prepared unique RunID; after connectivity returns re-audit all resources/path absence before launch.

# S5 current: Stage B local tests passed; awaiting fresh resource audit

Authority: `docs/S5_CUP_S4/USER_TASK_LAYER_SELECTION_SOURCE.md`, SHA256 `6eed7e9a239e171821ebe4c2db8ead3e1c623b933098401197ecbaef6c08c7a3`. The new task explicitly authorizes P3/P5 full-channel selection, stop-gradient class-drift CUP adaptation, and autonomous compatibility/numerical fixes within the stated boundary. Previous damage-definition confirmation wait is superseded.

Current implementation: isolated P3/P5 static-retention core with two observation sites and one retention sink; original S4 source/45 canonical files unchanged. Local 4 S5 contracts and inherited optional-metric-schema test PASS (5 total), compile PASS. Stage B protocol fixed before runtime in `configs/s5_layer_probe.json`: original TRAIN32, two independent batch16, original S4 / P3-noop / P5, five updates each, total30; no CUP, no victim, no official AP. RunID planned `s5_p3_p5_train32_layerprobe_seed0_20261018_v1`, UNLAUNCHED.

Next: RRCTL read-only fresh resource admission, exact-once diagnostic launch, actual module/ROI/GT/Fourier/Adaptive/no-op audit and paired layer evidence. Stage C and formal S5 remain NOT_RUN, no S5 config lock. P5 chosen only on clear repeatable evidence; otherwise P3 per user rule. After layer choice, implement/diagnose CUP, current-source GPU smoke, one new independent S5 review, original native AMP/budget/resource gates, then already-authorized formal run. See `docs/S5_CUP_S4/P3_P5_DIAGNOSTIC_PROTOCOL.md`.

## Prior S5 audit history (superseded waiting state)

# S5 current state: source audit complete; damage-definition decision pending

Authority: `docs/S5_CUP_S4/USER_TASK_SOURCE.md`, SHA256 `3f3521e3b82be7765ae35aa07850a01cbc12c8caf7c952233609f0e8f067947d`. New branch `codex/s5-s4-cup-adaptation-20261018`, based on S4 result-code commit `686eaacad069aac30f7fc7af4f10bebdeec1c390`. S4/S3 products are read-only.

Source audit: S4 actual Frozen L_ret is all-channel P3/model.15 GT-ROI feature drift, instance-mean reduction; its exact class baseline is pi_k=n_k/N. Existing class-drift diagnostics are P5 and cannot directly be reused. Source-derived CPU synthetic audit passed 5 cases: max scalar decomposition error 3.637978807091713e-12, max feature gradient difference 0, baseline correction no-op difference 0; all45 canonical source bytes remain exact. Real TRAIN32/model/Fourier/Adaptive equivalence and 5-step are NOT_RUN.

S3 CUP history/EMA/weight projection can be separated from CNLP, but its actual damage is post-virtual-learning clean-query detection-loss difference. Replacing it with same-input Frozen P3 drift (proposed minimal candidate A), or same-model GT detection-loss delta (B), changes this definition. The user's final-section stop rule requires this distinction to be reported before adopting it. No S5 loss was changed, no parameter was selected, no reviewer/GPU job/optimizer update/formal RunID was created. Formal seed0 24epoch/9144step + native-AMP victim200 is authorized only after the pending method definition and all correctness/resource gates are resolved.

Read `docs/S5_CUP_S4/S5_PARENT_FIDELITY.md`, `S3_CUP_DEPENDENCY_AUDIT.md`, and `CUP_S4_DESIGN.md` for the source audit and concrete proposal. Config lock/checkpoint/AMP/training/evaluation results for S5 are NOT_RUN. Do not inherit an old S4 monitor instruction as permission to launch S5 or rerun S4.

## Inherited S4 history (superseded by final S4 report and the S5 state above)

Current authority: execution/S4_CURRENT_STATE.json. Repair source5cd8158e8312c1fcebb9fa26ac551acc3cd3a669, native optimizer inherited; CPU7contracts passed, original45S0 source and frozenS4 method/map/config unchanged. Full authorization in docs/S4_AMP_RECOVERY_20261018/USER_TASK_SOURCE.md.

2026-10-18 05:42UTC three-windowRRCTL resource audit passed. Before launch, freshpreflight05:44:35UTC detected GPUcomputePID1649515 using26732MiB,util100%,free18426MiB. No ready/launch/binding or remote mutation occurred. Retained evidence execution/lifecycle/s4_dataset_reuse_cpu_audit_seed0_20261018_v1/20261018T054429711271Z/preflight.json. RunSpec first_step schema omission corrected before ready; all3specs now locally parse, no scientific code/config change.

Fresh fullCPUdataaudit, actualCUDA4trialsmoke, formalvictim200 remain NOT_RUN. Continue same15minmonitor when originalresourcegates pass. No otherprocess signals, duplicateRunID, noise regeneration, data/weight transfer, cleanup, environment change, secondreviewer or APtuning. OriginalS4noise/dataset preserved; original victim terminalfailure retained.

Latest monitor2026-10-18 05:49:48UTC: newthree-window check passed but finalpreflight detectedcomputePID1651450/26728MiB/util100%,free18430MiB; all4lifecyclepathsabsent, no ready/launch orbinding. Preserve resourcewaiting, no additionalexperiment orsignal. Evidence execution/lifecycle/s4_dataset_reuse_cpu_audit_seed0_20261018_v1/20261018T054943119942Z/preflight.json.

RRCTL ownership query 2026-10-18T05:54:07.517381Z confirmed PID1651450 (UID1017 liuyunf+, PPID1639842, python Test.py under GenPTW; GPU26740MiB) is not S4. Current RRCTL user guoqinglong. Evidence execution/GPU_PROCESS_OWNERSHIP_20261018.json. No signals/state changes. Keep waiting for GPU.

2026-10-18T06:20Z: stable three-window resource audit passed; RRCTL final preflight passed. Unique CPU-only read-only data audit s4_dataset_reuse_cpu_audit_seed0_20261018_v1 launched once, workload PID1656351/executor1656340, first-step health healthy. No GPU workload. Await RRCTL completion.

DATASET AUDIT COMPLETE: RunID s4_dataset_reuse_cpu_audit_seed0_20261018_v1 completed exit0; full fresh integrity/data/pixel audit passed: 16551/4952, 13256 Person GT changed, labels+clean validation byte-identical, non-target 10456 byte-identical, outside0/global8/overlap4, unique IDs, original45 source hashes, exact noise/integrity/manifest SHAs. Audit took87.64s, zero GPU forwards. RRCTL completion health->wait->pull artifacts3 passed. Local artifacts at C:\Users\20272\.codex\visualizations\2026\09\17\01a0afba-bca4-7582-815c-b2a5241bad4a\s4amp_small_20261018\s4_dataset_reuse_cpu_audit_seed0_20261018_v1. The next unique GPU stage is smoke, gated by a fresh three-window resource check.

ACTUAL CUDA SMOKE LAUNCHED: after fresh 3-window and final preflight gates, RunID s4_victim_amp_real_smoke_seed0_20261018_v1 launched once at2026-10-18T06:36UTC on source5cd8158e8312c1fcebb9fa26ac551acc3cd3a669. First-step healthy; stage native_S0; process PID1660639/executor1660631. Dataset re-audit passed first as recorded in docs/S4_AMP_RECOVERY_20261018/S4_DATASET_REUSE_AUDIT.md. Smoke is still running; no formal victim recovery launched.


LATEST UPDATE 2026-10-18T06:52:36.523097Z: the unique actual-CUDA AMP smoke `s4_victim_amp_real_smoke_seed0_20261018_v1` FAILED exit1 at 2026-10-18T06:37:38.790191Z in `native_S0`, after reaching its five-actual-batch boundary. The smoke raised `AssertionError: No actual native optimizer update observed` at the boundary callback. This means zero successful optimizer-step post-hooks were observed within five batches; per-batch GradScaler skip/backoff rows are missing from the bounded RRCTL pull, so native overflow versus observer accounting is unresolved. RRCTL transport retrieval of the small AMP CSV was denied before transfer; no raw SSH fallback. Dataset reuse audit passed. Paired equivalence, real S4 AMP trial, synthetic overflow trial, and formal recovery are NOT_RUN/NOT_LAUNCHED. Preserve failure; do not reuse RunID or claim algorithm/AMP result. Details: `docs/S4_AMP_RECOVERY_20261018/REAL_GPU_AMP_SMOKE.md` and `execution/S4_AMP_SMOKE_FAILURE_20261018.json`.


AMP LOG RETRIEVED 2026-10-18T07:58:27.295490Z: RRCTL transport fetched `native_S0/amp_audit/AMP_SCALE_AND_SKIP_LOG.csv` for smoke v1, 4,215 bytes, SHA-256 `f8cb06f598aa331e9f0f4cebeec2db216b3c3e0cc1cb8e0dc0627b59408ff79a`, five rows. Every row reports optimizer attempt observed, native scaler skip/backoff true, successful updates zero, raw loss/model/EMA/optimizer finite. AMP scale backed off 65536→32768→16384→8192→4096→2048. Thus the no-update assertion was caused by five consecutive native GradScaler skips within the fixed five-batch S0 control prefix, not a missing optimizer hook. The specific nonfinite gradient tensors and whether a later lower scale recovers were not retained/observed. The smoke boundary stopped before paired S0, real-S4 or synthetic overflow trials. Formal recovery remains NOT_LAUNCHED. Next bounded harness adjustment is restricted to allowing the paired S0 controls to complete when every zero-update row explicitly shows native skip+backoff; exact equivalence remains mandatory, and real-S4/synthetic trials still require actual successful updates. Keep the 16-skip ceiling, 58-batch cap, and formal args unchanged; use a new unique smoke ID.


LATEST 2026-10-18T08:24Z: S4 AMP smoke v2 passed after a narrowly scoped harness fix; original S0/repaired-S0 controls matched exactly, and actual-S4 plus synthetic-overflow recovery checks passed. Fresh resource gate passed and the authorized victim-only recovery RunID `s4_tfss_utility_seed0_20261017_v1_victim_recovery01` is running on source ab626917f20917e58cf30033ac99df00938cb561; first-step RRCTL health is healthy, PID 1696343. Preserve original noise/data; do not relaunch.


FINAL VICTIM RESULT 2026-10-18: recovery RunID `s4_tfss_utility_seed0_20261017_v1_victim_recovery01` completed 200 continuous epochs, saved and evaluated `last.pt@200` and `best.pt`, then the original pipeline exited 1 during comparison assembly with `KeyError('Recall_all')`. Historical S0-S3 summaries do not contain aggregate Recall_all/Precision_all. The remote RRCTL state remains failed; the training/evaluation outputs were independently hash-verified and small JSON/CSV files were recovered through RRCTL transport. Full S0/S1/S2/S3 and VOC20 report is in `docs/S4_AMP_RECOVERY_20261018/S4_S0_S1_S2_S3_FINAL_REPORT.md`; aggregate/per-class tables and pull provenance are adjacent. Comparison code now treats absent historical aggregates as unavailable, and a bounded failed-run artifact recovery tool is available under `scripts/recover_s4_failed_result_artifacts.py`. No experiment was restarted, and this single-seed result is descriptive only.
