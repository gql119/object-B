# S5 完整实验报告（主评价 last.pt@200）

S5 = S4-TFSS-Utility + CUP-S4(P3)。实验执行完整且通过验收；预注册的综合研究成功判据未通过。目标抑制更强，但非目标保持及最差类别损伤变差。

- 唯一 RunID：`s5_tfss_utility_cup_p3_seed0_20261019_v1`；运行源码 `e29d4662465159b5a1d1d7a5fd6c19c545f764e8`。
- 冻结配置 SHA256：`afdcd04e925d73f50eeb9d450bdbb259195d1500f61f1e46ecd08dd66807f0c8`。
- 完成时间：2026-10-19T08:48:43.187875+00:00（UTC）；未重新运行任何阶段。
- 完整扰动生成：24 epoch / 9,144 Adam 更新；CUP 9,144 更新；206 个 checkpoint 保留。
- 数据审计：16,551 train、4,952 clean val；6,095 目标图像、13,256/13,256 Person GT 改变；标签、clean val、非目标图像字节一致；框外 0、独占区域 ≤8/255、与非目标重叠区域 ≤4/255。
- Victim：随机初始化 YOLOv8n，seed0，200 epoch，batch36，workers16，AMP/native SGD；CSV 连续 1–200，last 内嵌 200 行逐值吻合。
- 主评价 last.pt@200，best 仅作次评价。

## S0–S5 主评价比较

AP 均为百分数；P/U/S、WCD/TCD3 也以百分数展示。NT 为其余 19 类宏平均；WCD/TCD3 是相对 clean 的带符号相对损伤，不是 AP 的百分点差。

|方法|Person AP50↓|NT AP50↑|NT AP75↑|全类 AP50|P↑|U↑|S↑|WCD↓|TCD3↓|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|S0|2.8993|69.1698|51.5898|65.8563|96.6595|90.2456|93.3425|28.1032|25.7851|
|S1|9.1670|70.4576|53.5902|67.3931|89.4382|91.9258|90.6649|27.0315|21.8793|
|S2|4.0368|68.9123|50.9453|65.6685|95.3490|89.9096|92.5494|30.8426|27.4399|
|S3|3.4414|68.9391|51.0290|65.6642|96.0350|89.9446|92.8901|26.9725|26.0754|
|S4|2.8159|69.3787|51.5251|66.0506|96.7557|90.5181|93.5330|29.5951|26.4482|
|S5|2.2586|69.0106|51.0207|65.6730|97.3977|90.0378|93.5733|31.3182|26.8352|

相对 S4：Person AP50 变化 -0.5573 个百分点；NT AP50 -0.3681 个百分点；NT AP75 -0.5045 个百分点。
WCD 变化 +1.7231 个百分点，TCD3 +0.3871 个百分点；二者均变差。

## 冻结判据

|判据|结果|
|---|---|
|Person_AP50_within_preregistered_limit|PASS|
|NT_AP50_above_literal|FAIL|
|WCD_below_literal|FAIL|
|TCD3_below_literal|FAIL|
|NT_AP50_improved_vs_actual_S4|FAIL|
|WCD_improved_vs_actual_S4|FAIL|
|TCD3_improved_vs_actual_S4|FAIL|
|NT_AP75_improved_vs_actual_S4|FAIL|

判定：`MIXED_OR_REJECTED_BY_PREREGISTERED_CRITERIA`。不能把执行验收通过、目标 AP 下降或 S 的微小上升解释为综合方法成功；本轮不自动调参、追加种子或重跑。

## S5 全 VOC20 指标

AP/Recall/Precision 单位为 %。Recall/Precision 来自原生 validator 的全局 best-F1 操作点；AP 使用完整置信度曲线，不能由 AP 推算 Recall。

|类别|AP50|AP75|AP50–95|Recall|Precision|AP50 vs S4（pp）|
|---|---:|---:|---:|---:|---:|---:|
|aeroplane|83.9666|66.3670|60.9025|73.3333|86.8962|+0.1849|
|bicycle|65.9045|27.8181|34.6214|54.1889|78.8442|-1.6792|
|bird|63.4333|41.7777|42.2986|57.8889|62.8586|-1.4653|
|boat|64.7846|43.1726|41.1372|56.2738|72.1566|-1.9995|
|bottle|49.6057|37.7413|32.0618|42.4309|69.5805|+1.5961|
|bus|82.0937|77.4007|71.1352|72.0676|85.9946|+0.4526|
|car|88.1467|74.3380|68.6066|79.1840|90.2748|-0.0334|
|cat|78.3020|65.5763|58.9982|75.4190|76.7525|-1.9942|
|chair|52.8277|38.7979|34.5421|45.2633|64.6630|+0.5351|
|cow|72.8246|59.0247|53.1788|68.0328|65.7953|+2.8756|
|diningtable|68.2100|45.6718|45.2675|56.3107|76.0041|+2.2165|
|dog|55.4116|44.5143|40.9770|69.1207|39.6196|-1.3901|
|horse|74.3582|42.7569|43.6271|71.2256|70.6440|-0.4310|
|motorbike|71.9218|34.5906|38.2457|68.3077|72.0168|-3.1262|
|person|2.2586|0.7651|0.9813|0.0221|32.3677|-0.5573|
|pottedplant|42.1838|16.5220|20.6167|33.1250|68.8249|-1.4424|
|sheep|71.0755|59.0328|53.4144|65.2893|70.5400|-2.4973|
|sofa|69.7676|60.1251|54.6559|69.8745|63.5015|+0.3075|
|train|84.7778|70.8288|61.8450|79.7872|86.6692|+1.0896|
|tvmonitor|71.6050|63.3362|53.8704|60.3896|81.5363|-0.1938|

完整各方法 20 类 AP50/AP75/AP50–95/Recall/Precision 见 [S5_VOC20_PER_CLASS.csv](S5_VOC20_PER_CLASS.csv)；完整聚合见 [S5_COMPARISON.json](S5_COMPARISON.json)。历史可选全类 Recall/Precision 缺失项保留缺失，没有补造。

## 不利类别与次评价

相对 S4 有 11 个非目标类别 AP50 降低：bicycle, bird, boat, car, cat, dog, horse, motorbike, pottedplant, sheep, tvmonitor。最差相对损伤类别为 dog。
best 次评价 Person AP50=3.2527%，NT AP50=69.9041%；不替代 last 主评价。

## 八项研究回答

1. 保持层选择 P3；固定 TRAIN32 证据中 P5 的代理改善混合、对抗保护梯度更大。不是普遍层优越性结论。
2. 层选择依据来自真实梯度、GT/尺寸、5-step 对照，不使用最终验证 AP 选择层。
3. CUP 使用 detached P3 类别漂移，固定校准，β=.95/T=1/边界 .5–2，经归一化分配原 L_ret 类别权重；实际权重与梯度见 CSV。
4. S4→S5 的设计变量为 CUP 类别分配；TFSS-U/.85、S0 损失与其权重、Fourier、TSFL、Frozen/Adaptive、SafeBG、预算及 Victim 协议保持冻结。一次结果不构成机制因果证明。
5. Person AP50 较 S4 更低，目标抑制更强。
6. NT AP50/AP75 较 S4 更差，未支持非目标保持改善。
7. WCD/TCD3 均较 S4 更差，最坏类别效用未改善。
8. 当前完整 seed0 结果不支持将 S5 判为通过综合标准的主方法；保留负面证据，不自动追加实验。

## 核验证据与边界

- 37 个原始小型产物逐文件大小/SHA 校验；全部逐类数组/已有聚合、带符号 P/U/S/WCD/TCD3 在本地重新计算一致。
- 远端 CPU 只读载入最终 noise/last/best，核验 Adam、scheduler、CUP、Fourier、RNG、有限性、权重身份与 200 行训练记录；零 GPU forward、零新优化更新。
- config/review 快照是流水线重序列化 JSON：原文件 SHA 与快照 SHA 分别记录，解析内容完全相等；未改动冻结配置。
- Frozen/Adaptive 原始权重 SHA、初始 state hash 与运行结束未改变标记一致；进程退出后未独立取得活体最终 teacher 参数 hash，不将该标记夸大成第二次参数审计。
- SSIM 未运行（缺少已安装工具）；LPIPS 未运行（缺少已验证权重绑定）；不填估算值。
- 单种子描述结果；没有显著性、收敛性、跨种子鲁棒性或背景吸收的因果结论。
- [LOCAL_ACCEPTANCE.json](LOCAL_ACCEPTANCE.json) 为主代理验收；[PIPELINE_LOCAL_ACCEPTANCE.json](PIPELINE_LOCAL_ACCEPTANCE.json) 保留流水线验收；[MAIN_AGENT_CHECKPOINT_METADATA.json](MAIN_AGENT_CHECKPOINT_METADATA.json) 为 checkpoint 证据。
- [ARTIFACT_PROVENANCE.json](ARTIFACT_PROVENANCE.json) 记录发布文件来源；大诊断/逐图 ledger 本地原样保留，权重与数据仅在服务器保留。

Git 发布采用现有 core.autocrlf 的 CRLF→LF 归一化；RAW 原文件未改动。ARTIFACT_PROVENANCE 中的 SHA 为原始取回字节；PUBLICATION_BLOB_HASHES 分别记录暂存 Git 字节与本地字节，已确认仅换行不同、解析 JSON 完全相等。
