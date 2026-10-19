# S5 最终分析

## Change
S5=S4+CUP-S4(P3)，唯一运行 s5_tfss_utility_cup_p3_seed0_20261019_v1。运行源码 e29d4662465159b5a1d1d7a5fd6c19c545f764e8；冻结配置 afdcd04e925d73f50eeb9d450bdbb259195d1500f61f1e46ecd08dd66807f0c8。保持 S4 空间公式/eta.85、S0 保护损失与权重、Fourier/TSFL/Frozen/Adaptive/SafeBG/预算及随机 Victim200 协议。

## Result
24/9144 noise、16551train/4952cleanval、13256 PersonGT、标签/框外0/8-4预算审计通过。Victim CSV 连续1–200，last内嵌200行逐值吻合，last/best身份与模型有限性通过。37个产物大小/SHA核验、逐类AP50/AP75/AP50–95/Recall/Precision已有聚合及带符号P/U/S/WCD/TCD3复算通过；CPU只读元数据审计0GPUforward/0新更新。

S5 last：Person .02258628164286698、NT .690105689344675、NTAP75 .5102067823101584、WCD .31318165930240155（dog）、TCD3 .26835235162192933。相比S4目标-0.5573pp，NT-0.3681pp，NTAP75-0.5045pp，WCD/TCD3更差。11个NT类别AP50降低；详见全量CSV。

## Finding
SUPPORTED：执行完整且通过验收。MIXED：目标抑制更强，非目标效用保持未改善；综合预注册判据未满足（MIXED_OR_REJECTED_BY_PREREGISTERED_CRITERIA）。不从AP推断背景吸收，不作显著性/跨种子结论，不以best替代last。

快照经JSON重序列化，与原锁定文件字节SHA不同但解析内容完全一致；原source/receipt SHA分别核验，未改变实验配置。SSIM/LPIPS依然NOT_RUN，历史可选Recall_all/Precision_all缺失不补造。运行结束Frozen/Adaptive不变标记与原文件/初始state核验保留，未把标记夸大成活体最终参数独立重哈希。

## Next
完成已授权的当前codex分支代码/小结果归档发布，并关闭唯一监控。无需再轮询已完成远端；不追加种子、调参、重跑或新方法。

## Evidence
- docs/S5_CUP_S4/final_20261019/LOCAL_ACCEPTANCE.json
- docs/S5_CUP_S4/final_20261019/MAIN_AGENT_CHECKPOINT_METADATA.json
- docs/S5_CUP_S4/final_20261019/S5_FINAL_REPORT_ZH.md
- docs/S5_CUP_S4/final_20261019/S5_COMPARISON.json
- docs/S5_CUP_S4/final_20261019/S5_VOC20_PER_CLASS.csv
- 原始37文件：BASE/s5_small_20261018/s5_tfss_utility_cup_p3_seed0_20261019_v1；SHA清单保留RAW_ARTIFACT_MANIFEST.json。
