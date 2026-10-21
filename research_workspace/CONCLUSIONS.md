# 当前证据 — 2026-10-17（旧待批准/NOT_RUN文字仅历史）

- C007 | decision | SUPERSEDED：原未批准RMS修订已获直接人类确认，abs1e-7/rel1e-4双门实施；608个完成配对通过。
- C008 | finding | SUPPORTED | bounded correct implementation：两batch各5步S0/no-op最大差0；一次独立审查与小GT主代理闭合、59检查点证据保留。不是完整实验成功。
- C009 | finding | SUPPORTED | bounded target proxies：C/U预注册GateB通过；U梯度更近S0，C较同强度Uniform保留高保护区域算子能量。无AP/收敛/不可学习性保证。
- C010 | finding | MIXED | NT utility：C的L_ret不如Uniform-C但GT signed损失变化更低；U GT代理未优于S0、完整Uniform比较缺失，不能选正式候选。
- C011 | finding | SUPPORTED | control reachability failure：Uniform-U batch1 step5候选RMS超出自身alpha≤1上限，控制器failed/exit1。原异常未保存具体图/幅度；不同轨迹解释为代码推断，未作新GPU复现实验。
- C012 | decision | OPEN | INSUFFICIENT_EVIDENCE：只59/60比较更新。完整TFSS未授权；停止自动GPU重试，任何新控制协议先明确范围并获人类确认。

证据均在docs/tfss/short_comparison_20261017及三份最终报告。

---

## 历史记录（以前的 NOT_RUN / 等待段落已被上文取代）

# TFSS research conclusions

- C001 | decision | ACTIVE | scope=tfss.short_protocol: Human authorizes isolated code snapshot publication and train-only C/U/pairedUniform short comparisons, after resource/correctness gates. Source docs/tfss/USER_TASK_SOURCE.md. Formal TFSS is not authorized.
- C002 | execution | OBSERVED | scope=tfss.source: Snapshot a5aefbb is pushed/reread; S0 exact45 source restored. Evidence docs/tfss/S0_RESTORATION_AUDIT.json. Historical runtimeGitSHA remains unknown.
- C003 | finding | SUPPORTED | scope=tfss.component_correctness: Eight CPU numeric contracts pass, including actual RMS/image clamp and detached scaling/backward preservation. Evidence docs/tfss/CPU_VALIDATION.json. Bounded component evidence only, not real VOC/GPU mechanism admission.
- C004 | hypothesis | OPEN | scope=tfss.efficacy: C/U may retain S0 target protection while improving NT utility; no 5step or victim data exists. Current recommendation INSUFFICIENT_EVIDENCE, no AP claim.

- C005 | finding | SUPPORTED | scope=tfss.replay: deterministic paired CUDA control repair yields actual canonical/noop maxdiff0 at one update/first16; not a full smoke pass. Evidence execution/tfss_smoke_v2_observation_20261017.json.
- C006 | finding | MIXED | scope=tfss.uniform_numeric_gate: actual GPU Uniform strict RMS gate failed; CPU saved-delta replay found11/16 targets not bracket-matchable within1e-9 in current float32 arithmetic. No matched-control efficacy conclusion. Evidence docs/tfss/RMS_GATE_BLOCKER_20261017.md.
- C007 | decision | PROPOSED | scope=tfss.numerical_protocol: closer-bracket solver plus BOTH abs1e-7/relative1e-4 RMS gates is unapproved. Existing1e-9 stays locked; WAIT_FOR_HUMAN, no automatic retry/reviewer/short/full run.

### C001

<!-- research-memory:f2112ee8745f61f83125f3be:C001 -->
Type: decision
Status: ACTIVE
Scope: ablation.s4_tfss_full.support
Effective at: 2026-10-21T15:05:51.920120+00:00

Independent S4_TFSS_FULL_OVERLAP changes only Person-NT intersection support .5 to1, keeps TFSS eta.85 and frozen S4 science/protocol, and waits for predecessor robustness full acceptance plus conflict-free scheduling before GPU smoke/review/formal.

Source: Ecc7738e5ecf131a2ed1baacd
Source role: user
Source ref: docs/S4_TFSS_FULL_20261021/USER_TASK_SOURCE.md
Occurred at: 2026-10-21T15:05:51.920120+00:00
Received at: 2026-10-21T15:05:51.923112+00:00
Evidence: 见来源事件

> 用户来源摘录：# Codex 任务：S4-TFSS-Full 全目标框空间调控消融实验
>
> ## 1. 研究目标
>
> 当前项目：
>
> Class-Selective Unlearnable Examples for Object Detection
>
> 正式基线：
>
> S4 — Safe-Soft05 + TFSS-Utility
>
> 本次需要建立一个独立消融版本：
>
> **S4-TFSS-Full**
>
> 研究问题：
>
> 原始 S4 在 Person 与非目标 GT 重叠区域设置固定 0.5 区域系数，使重叠区扰动上限从 8/255 降为 4/255。
>
> 本次希望取消这一固定重叠区域减半机制，让整个 Person GT 框内均采用 8/255 基础扰动预算，并完全通过原有 TFSS-Utility 权重进行位置相关的扰动缩放。
>
> 需要验证：
>
> 1. 去掉 4/255 重叠区限制后，是否进一步降低 Person AP50。
> 2. TFSS 能否在没有重叠区固定减半机制的条件下，保持非目标类别 [已截断，按来源继续读取]

### C002

<!-- research-memory:e34f88c7daadb7cb208edddb:C002 -->
Type: finding
Status: SUPPORTED
Scope: ablation.s4_tfss_full.cpu_components
Effective at: 2026-10-21T15:03:17.119020+00:00

27 CPU tests passed; exact component parity/gradient checks and PNG8/outside0, formal GPU/AP still NOT_RUN.

Source: Eec8d9a06d88ea409afb17751
Source role: document
Source ref: research_workspace/analysis/2026-10-21-s4-tfss-full-preparation.md
Occurred at: 2026-10-21T15:03:17.119020+00:00
Received at: 2026-10-21T15:04:01.757860+00:00
Evidence: research_workspace/analysis/2026-10-21-s4-tfss-full-preparation.md
Protocol: CPU component parity/export/cache only
