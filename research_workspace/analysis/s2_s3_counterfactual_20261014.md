# S2/S3 代码准备

## Change

用户 2026-10-14 “先写相关代码”授权在 S1 尚未完成时准备独立本地实现。建立 `codex/safe-soft05-s2-s3-counterfactual-20261014`；独立 CNLP 双路径、共享 query assignment、CUP 历史聚合、outer loop/checkpoint、train-query manifest 和父版本准入代码。S1、S0 历史源文件均未更改。

## Result

55 项 CPU 测试通过，47 新测试和8历史契约；native YOLOv8n合成输入 J=1/3/5 的虚拟更新梯度链可达，有限差分与 autograd 最大误差1.97e-11；synthetic outer-loop continuation 的参数/Adam/scheduler/RNG/CUP/CSV一致。源码状态为本地未提交修改；详见 docs/S2_S3_CPU_VALIDATION.json。

## Finding

SUPPORTED（仅正确性）：当前 synthetic CPU 范围的梯度链、随机状态/BN隔离、历史聚合和checkpoint恢复。OPEN：真实 NT GT覆盖/损失尺度、稀有类稳定性、CNLP权重/方向、GPU显存耗时、生产 Parent 集成及最终检测效用。CPU 通过不等于科研有效、收敛或正式准入。新任务母版 SHA182a145937a888d489eaf232ca7229a0c252c46626f4cec65d2aea4581b50311。

## Next

等待 S1 完整归档，按严格1pp双条件锁 Parent，绑定真实预算/source/config/reference身份。仅用 train开发数据完成CNLP/CUP审计和标定，锁定两实验协议，再独立顺序S2→S3。Parent、J、lambda_CNLP、CUP参数未选；真实VOC/GPU准入/正式noise/victim均NOT_RUN；不得上传未完成实验的代码至GitHub。


## 2026-10-15 已核验推进

Change：完整S1归档后按用户原规则锁S0，证据parent_selection.json和docs/s1_archive/。新增有界train-only J1/3/5标定与后续生产循环准入入口草案；原45个Parent文件字节不变。CODE-004修复EOT记录缺口，只记录mean-vector后norm/cos，旧v1证据全部保留。

Result：J1v2 38case/76EOT，215.592秒，peak5,961,734,656bytes，median clean learning gain=7.724761962890625e-5，21/38非零平均CNLP梯度，强attack-proxy冲突0/21。J3v2同计划，323.912秒，peak7,164,462,080bytes，median gain=8.499622344970703e-5，23/38非零梯度，强冲突0/23。全部zero outer updates、Frozen/Adaptive/Fourier未改变；J5仍待结果。此处attack仅BG_sel+rel_sel的梯度代理，不包含Parent Adaptive target项，不能宣称完整attack梯度无冲突。

Finding：SUPPORTED：S1全流程和预注册S0选择；bounded train signal/梯度可达。OPEN：新模块最终检测效用、GT边界覆盖、真实生产恢复、CUP真实sink与正式准入。J1/J3数值不构成最终收敛或AP改善证据。历史S1存在12个native allGT未覆盖但合法NTGT，新增显式CNLP边界审计，不能假定S1 rematch自动存在于CNLP。

Next：完成J5和预注册选J/权重；原Parent生产等价、合法重叠NTGT覆盖、S2/S3真实5步/续跑/像素检查；最终candidate一次科学审查、两配置锁定后顺序完整实验。真实GPU准入未完成，尚无allow_run，不推送未完成实验。
