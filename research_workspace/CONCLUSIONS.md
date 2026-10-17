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
