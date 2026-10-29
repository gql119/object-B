# Original S4 AT 科学等价加速与条件重启报告

决策：**CONTINUE_ORIGINAL_AT**。条件重启授权已记录，尚未触发。完整GPU性能与科学等价结论：**INCONCLUSIVE / INSUFFICIENT_EVIDENCE**。

## 1. 授权与实际运行

本次人类授权允许通过验证且总耗时有净收益后安全停止并从epoch0重训；不需要再次确认。授权原文为 [USER_AT_SCIENTIFIC_SPEEDUP_AUTHORIZATION_20261029.md](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/docs/S4_AT_ROBUSTNESS_20261027/USER_AT_SCIENTIFIC_SPEEDUP_AUTHORIZATION_20261029.md)，SHA256 `91199b1ea8b5d39dd66ca1f52b68a122233349889a548ee82efde35f62bb5197`。

实际运行仍为 `s4_at_serial_seed0_20261029_v2` / `S4AT4_Clean_seed0`，源码 `51ce790b7de30ea032c7c4bf7c85a8f530f3fcfb`。最后核验 2026-10-29T15:09:27.160038+00:00：epoch21、已完成20轮、9515batch、5195SGD、6次AMP跳过，UID/PID/start均一致。尚无完整200epoch组被验收。原唯一监控器保留，COCO不变。

## 2. 瓶颈与原始耗时

正式完整batch均值 1.6884s，PGD结束到batch progress平均 0.1302s（联合外层和审计，不是纯外层GPU时间）。最近5个完成epoch均值 785.34s；控制台训练约12:29–13:12、验证7.7–13.4s。PGD单步、纯DataLoader等待、独立SGD和同步开销均未被现有日志分别计时。

GPU均值 83.5%、85–86°C，6/8热限制和4/8功率限制Active；CUDA真实分配峰值 8990.74MiB。CPU训练树约1.18核，RAM约71.50–73.86GiB，I/O wait 0.0132%，无持续换出；本窗口其他UID CUDA compute-app为0。证据支持PGD计算和同步审计路径为主要方向，不支持以低显存为由增batch/并行。

## 3. 新旧实现完整差异

参考是实际运行51ce790的PGD源码，SHA `63bfa27086953566efb771e8bddf4aba5cc7363b8198dd3d73a28ba93df08578`。隔离候选02仅将诊断loss/parts/norm的逐值CPU回读汇成PGD末尾一次回读；候选03再向量化诊断用匹配GT计数。模型、动态TAL、GT、FP32内层、10次梯度、随机扰动、8/4预算、投影、即时守卫、外层AMP/SGD、checkpoint代码均未替换。

完整AST差异：[CANDIDATE_02_AST_DIFF.patch](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/CANDIDATE_02_AST_DIFF.patch) 与 [CANDIDATE_03_AST_DIFF.patch](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/CANDIDATE_03_AST_DIFF.patch)。候选仅在execution隔离目录；没有新的正式源码SHA/Release/RunSpec，也未发布为可运行正式候选。

## 4. 数学与数值验证

两候选各5个本地CPU案例全部PASS，atol=rtol=0。逐步PGD输入/梯度/loss、完整TAL输出、最终扰动、所有审计字段、实际外层SGD/参数梯度、BN、EMA、optimizer、scheduler一致。控制状态恢复后的下一PGD也完全一致。CUDA_initialized=false。

这仅覆盖真实fixture CPU batch2/160/AMP关闭；正式batch36/640/AMP、真实小目标、多batch生产恢复和实际CUDA完整状态仍待测。CUDA AMP跳步等价未被CPU disabled-scaler结果代替。原正式6次AMP跳过占4943尝试的0.1214%；4055个无尝试batch来自原生accumulate2。

## 5. 真实性能对照

| 指标 | Old实测 | New实测 |
|---|---|---|
| 正式完整batch墙钟 | 1.6884s均值 | 未测量 |
| 同固定输入、充分预热GPU基准 | 未执行 | 未执行 |
| PGD纯GPU秒/batch | 未独立计时 | 未测量 |
| 外层纯GPU秒/batch | 未独立计时 | 未测量 |
| 新旧真实性能提升 | null | null |
| CUDA等价/AMP/恢复 | 原实现已有正式前验收 | 新候选未完成 |
| 编译、初始化新增成本 | 无新编译 | 未测量 |

未将CPU验证进程的21–24秒运行时间比较为加速数据，也未在有当前正式GPU竞争时运行基准。未申请第二个GPU controller。

## 6. 继续与重启成本

当前4组总计划800epoch/368000训练batch，已执行9515batch，尚余358485batch及后续验证。以最近5轮实测均值直接外推，方案A剩余约 **170.0小时**；若使用已观察epoch总耗时的最低/最高值作敏感性范围，约164.0–183.5小时。这不是保证ETA：后续模型、温度、close_mosaic及验证成本会变化。

方案B必须从随机YAML seed0统一重做受影响配对组；当前没有已完成组可以不经证明混入新实现。新旧GPU速度比、基准/审查时间、编译/初始化、合法停止/启动成本均缺失，因此 **B总耗时=null，净节省=null**。

计算规则：`T_A=剩余训练+后续验证/保存/最终评价`；`T_B=需重做的全部训练/实测速度比+验证/保存/最终评价+基准/启动/停止成本`。须同时满足完整科学门槛、稳定实测吞吐增益至少15%、包含不确定性的净节省至少10%。尚不能代入一个猜测速度比批准重启。

## 7. 重启门槛逐项状态

| 门槛 | 状态 |
|---|---|
| 人类条件操作授权 | 已具备 |
| 当前旧运行身份与封存checkpoint | 已核验；epoch19 safe optimizer boundary=true |
| 小CPU固定输入科学比较 | 两候选各5例零差异PASS |
| 当前源真实CUDA/AMP/nativebatch36/恢复 | 未完成 |
| 无竞争GPU稳态Old/New基准 | 未执行 |
| 真实性能增益至少15% | 未证实 |
| 重训净收益至少10% | 未能计算 |
| 可验证的非终态暂停及原控制器继续 | 不具备现成接口/证据 |
| 新版本一次科学审查与主闭环 | 未进入；禁止在CUDA前送审 |
| 新不可变正式Release/RunSpec/READY | 未创建 |

`RRCTL resume`只恢复本地索引，不恢复训练。现有`abort`含TERM后SIGKILL兜底，且没有合作式epoch暂停；旧queue对既有输出/锁不能原样重入。workers16预取逐位恢复标记false。已验收的受控resume证据不能冒充任意时刻暂停/继续整条队列的证明。因此此次不先停旧任务获取测试窗口，不修改准入规则，不通过并发基准干扰现行AT。

## 8. 本次真实操作与后续

已执行：只读身份/资源/日志/源SHA审计、封存checkpoint CPU元数据检查、两项隔离候选及10个合计CPU案例验证，生成审计和测试合同。未执行：GPU benchmark、新候选科学审查、停止、重启、正式替换、修改旧checkpoint或Poison数据。

候选一旦有安全独占GPU窗口，需按 [PENDING_CUDA_BENCHMARK_CONTRACT.json](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/execution/speed_audit_20261029/PENDING_CUDA_BENCHMARK_CONTRACT.json) 完成真实等价与A/B性能，再闭合一次科学审查、成本与控制恢复条件。没有满足条件前，当前四组仍使用同一原实现严格串行。当前选择 **CONTINUE_ORIGINAL_AT**，不是宣称优化没有收益或已经达到15%。

详细原始证据和代码审计见 [S4_AT_SPEED_OPTIMIZATION_AUDIT.md](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/docs/S4_AT_ROBUSTNESS_20261027/S4_AT_SPEED_OPTIMIZATION_AUDIT.md)。


## 后续控制工程进展（2026-10-30香港时间）
原先“无已验证暂停通路”的诊断现已部分推进：独立CPU04真实RRCTL探针验证了仅保持队列监督器、不暂停科学子进程，并在子进程自然结束后释放同一监督器的机制。当前正式队列未操作，CUDA/速度仍未测量；详见 [S4_AT_CUDA_BATCH_OPTIMIZATION_AND_RESTART_REPORT.md](C:/Users/20272/.codex/visualizations/2026/09/17/01a0afba-bca4-7582-815c-b2a5241bad4a/s4_at_robustness_20261027/docs/S4_AT_ROBUSTNESS_20261027/S4_AT_CUDA_BATCH_OPTIMIZATION_AND_RESTART_REPORT.md)。旧报告原始测量结论保留。
