# COCO S4 Bug 修复记录

修复前审计40f5890先于本轮科学代码修改；当前计算候选0113b9b65b0aba4b0132ff5b1f4f58782756cc53。9项确认问题中，7项属于管线/记录/准入，2项属于验证器（S4-008/009），不混称9项数学错误。

- **S4-001 — Canonical TFSS optimization missing**：CLOSED_REAL_CPU_CUDA。Restored original S4Trainer + SoftSpatialProjection + CleanMapCache + optimize_s4;2/2 actual Frozen/Adaptive sinks per run
- **S4-002 — Wrong optimizer entry bypassed native persistence**：CLOSED_NATIVE_CHECKPOINT_RNG。Actual64x3 completed-update state/Adam/scheduler/RNG verified; original final16x3 pack validated. Old unsaved state remains unavailable
- **S4-003 — Exporter demanded absent num_classes metadata**：CLOSED_REAL_PARAMETER_EXPORT。Original native pack validator checks actual bound COCO80 model/config; actual savedpack→generator→PNG passed CPU/CUDA
- **S4-004 — Exporter claimed CUDA from counter/availability**：IMPLEMENTED_ADDITIONAL_EXACT_HOOK_PROBE_WAITING_RESOURCES。Full S4 actual raw CUDA44forwards proved; additional exact export callback CPU-negative/CUDA-positive check WAITING_RESOURCES
- **S4-005 — File SHA compared to tensor-state SHA**：CLOSED_CONTRACT_TEST。Canonical state digest matched; real production victim/evaluation dependency validators accept good synthetic contracts and reject mismatched initialization
- **S4-006 — CPU helper initialized CUDA RNG**：CLOSED_REAL_CPU_ISOLATION。Adapter CPU-only context preserves original source; actual learnedTeacherCPU16 passed with no-init guard and empty native CUDA RNG; CUDA original RNG unchanged
- **S4-007 — Superseding invalidation/no-formal scope not consumed**：CLOSED_LAUNCH_GUARDS。5 formal prepare/launch/READY guards reject; explicit validation-only manifest bound exact latest human-task SHA
- **S4-008 — 433 observer compared preEOT identity at postEOT/letterbox**：CLOSED_REAL_FROZEN_CHAIN。Passive originalEOT→originalletterbox→actualFrozeninput identity chain recorded; actual Adaptive weighted inputs recorded
- **S4-009 — New test compared stochastictraining with deterministicexport**：CLOSED_REAL_CPU_CUDA。Original generation context used on both sides only in validation, training randomscale/dropout preserved; failed probe retained

保留失败记录：433仅toy组件；第一次合同检查遇BOM读取错误，修正检查器的读法；临时core RNG改动触发原始哈希锁后完整恢复原文件，改为适配层CPU隔离；2145hk真实CPU因验证上下文错误FAILED，保留完整状态、参数/checkpoint和退出身份，未改为通过。其后的新2215hkCPU与2230hkCUDA都是真实独立新RunID/exit0。

准确的源码锁范围：原始锁8文件中7个逐字节不变，Safe-BG使用已授权维度差异；b2_semantic的已授权适配另有差异清单。不能声称8个原文件全未改变。当前修复未新增科学定义变化。

本轮绝未启动24epoch正式Fourier、Export25K、Clean200、S4_200或正式评价。下一次正式运行必须后续明确人类授权、新RunID/源与配置绑定、原始初始化；禁止从错误训练续训。
