# TFSS-Utility：第五步失败恢复与同扰动静态对照

日期：2026-10-17。状态：本轮六项诊断任务完成；建议开展一轮完整 Utility 验证，但本轮没有启动或授权完整实验。

**判断：Utility 在现有固定 TRAIN32、相同基础扰动和相同实际 Person RMS 下，表现出比均匀缩放更合理的空间选择性。目标代理保持门通过，NT 平均代理有小幅正向信号，值得做一轮完整效用验证。它还没有证明最终 Person/NT AP 更好。** 逐 GT 结果并不一致，不能用平均改善替代完整 victim 评估。

## 1. 第五步失败的直接原因

从原 batch1 Uniform-U 的 step4 检查点恢复 Fourier 系数、Adam、scheduler、Python/NumPy/Torch/CUDA RNG 和 active frequency indices，按原 FP32 路径确定性重建 step5/EOT1 的基础扰动。原失败时没有单独保存这一时刻的原始 delta；本轮保存的是**从原检查点重建的张量**，不能称作原运行现场已落盘的 delta。恢复后原异常逐字复现，尚未执行新的 optimizer update。

第一个且唯一超过该步可达性门的图像是 batch1 index10（全局固定样本 index26），`2010_004838`：

| 量 | 实测值 |
|---|---:|
| 原 Utility 目标 Person RMS | 0.007848653708569487 |
| 独立 Uniform-U 基础扰动在 alpha=1 的最大实际 RMS | 0.007848435978964851 |
| 目标超过最大值 | 2.1772960463566082e-7 |
| 相对缺口 | 2.7741013009394795e-5 |
| 同 Utility 基础扰动在 alpha=1 的 RMS | 0.007848653708569487 |
| Utility 原记录目标重放最大差 | 0 |

原错误为 `Uniform target exceeds own S0 maximum: no silent amplification`。锁定门要求绝对误差不超过 1e-7 **且**相对误差不超过 1e-4；这里相对门可以通过，绝对门不能通过。把独立控制的 alpha 限定在 [0,1] 时，该目标不可达。

该图只有 Person、没有 NT GT，原 clean Rnt 为 0，Utility 的 W 为 1，因此目标正好等于 Utility 自身未缩放基础扰动的最大 RMS。两个独立优化轨迹已经产生不同 Fourier 系数：最大系数差 0.005097646266222，重建基础 delta 最大差 0.01576470583677292。其他 batch 图像产生的梯度也会更新共享 Fourier 参数；所以即使这张图当前 W=1，两条历史轨迹也不需要一致。相同 RNG 和 active indices 均已核验。

恢复的 Adam step 为 4，lr=0.001，scheduler.last_epoch=0；Fourier/Adam 状态有限，恢复前后优化器摘要相同，且 CPU 核验与原 step4 的系数及优化器状态完全相同。**直接原因是独立轨迹导致本图控制自身 RMS 上限略低于目标，不是 NaN、优化器损坏或本图空间缩放失效。** 这只解释本次失败，不证明整段优化效果。

恢复张量留在服务器新诊断输出，未传回本地：`RECOVERED_FAILURE_STATE.pt`（SHA e61531b7c34b20b256c5c973d1a92d4bc170b389a825c5bb788d36121972e172）及 `RECOVERED_UTILITY_STEP5_BASE.pt`（SHA 5b5370800ff48949574426743b5990f32d02defd2e09d8b355c91e74f4327901）。输出根、基础 delta 摘要、完整恢复元数据见下方证据链接。

## 2. 两类对照分别回答什么

| 对照 | 基础扰动 / 优化状态 | 回答的问题 | 本轮状态 |
|---|---|---|---|
| 原独立优化轨迹 | 各方法各自 Fourier/Adam 历史 | 整个优化过程能否持续配对、最终短程效果如何 | 保留原失败：59/60 更新；Uniform-U batch1 step5 未完成 |
| 本轮静态同扰动对照 | 逐一读取原 Utility 同一 pre-step 状态，所有臂共享基础 delta/输入/模型/EOT 随机抽样 | 固定扰动状态和总强度时，空间 W 是否优于均匀 alpha | 完成两批全部五个状态、EOT2，共 320 个逐图 RMS 配对 |

本轮三臂为 `same_state_S0`、`Utility_replay`、`Static_Uniform_U`。这里 same_state_S0 是 **Utility 的 Fourier 状态下原 S0 扰动算子**，并非历史独立优化 S0 的最终扰动。静态 Uniform 使用同一基础扰动，只以逐图 alpha∈[0,1] 替换空间 W，并匹配实际 image clamp 后的 Person RMS。Utility 保持 `W=1-0.15 Rnt(1-Qt)`。

读取原 pre-step1..5，每个状态测 EOT1/2：共 60 条 batch 损失/梯度 forward 记录、320 个静态 RMS 配对。未执行 optimizer.step 或 scheduler.step，新增 Adam 更新数为 **0**。最大 RMS 绝对误差 7.2966463296811335e-9，相对误差 1.571834296159074e-6，均通过原已批准双门。没有放宽门、放大 alpha、修补像素或改变独立轨迹。

**缺失的同扰动 Utility 对照已经补齐，包括 batch1 第五状态；原独立轨迹的第五次更新仍未完成。** 两者不能合并记成原实验完整 60/60。

## 3. 损失、参数和日程保持核验

原 S0 的 45 个源码文件 SHA 不变；TFSS-C/U 原实现、损失、eta、Adam、Fourier 初始化、epoch0/warmup、EOT2、batch16、预算 8/4/0 和图像 clamp 均不变。没有重训 S0/C/U 或修改其保存状态。32 张训练开发图、65 个 Person GT、46 个 NT GT 与原 manifest 一致；没有使用 clean validation 优化扰动。

Utility 的所有 20 个状态/EOT 单元，原损失/RMS 重放最大差为 0，保护项 Fourier 梯度向量与原保留张量逐元素完全一致（最大差 0）。30 组模型/Fourier/Adam 前后检查均不变，输入检查点和 clean maps SHA 不变。

新静态诊断源提交为 `7cdcb382877496ee8be59b9099ede35fde672a56`，仅增加独立诊断入口、分析、协议及测试。当前源 smoke 通过后，执行了一次针对**本轮新静态对照实现**的独立科学审查，结论 scientifically_correct/allow_run；这不改写旧 TFSS 审查，也不把自检冒充独立通过。三个新 RunID 均完成 exit0：

- `tfss_utility_static_smoke_seed0_20261017_v1`：当前源 smoke；canonical/no-op 最大差 0。
- `tfss_utility_static_measure_seed0_20261017_v1`：完整固定状态测量，451.5264 秒，0 个新更新。
- `tfss_utility_static_analysis_seed0_20261017_v1`：输出分析/绘图，CPU、0 个模型 forward/更新。

三个 CPU 契约测试通过；独立审查也实际复跑通过。最初一次 unittest 模块路径调用误入已安装的同名 tests 包，随后使用项目 discovery 正确执行；没有把那次启动错误记为测试通过，没有修改环境。五张新派生图已逐张视觉检查。数据、权重、原始 RGB、恢复张量均未下载；仅拉取小 JSON/CSV 和派生图。本轮没有 GitHub push。

## 4. 同总 RMS 下的空间选择性

下表是实际 image clamp 后的扰动**能量保留比例**，参照同一 Utility 状态的未缩放基础扰动。先在每张图内汇总五状态/EOT，再在有效图间平均；不是把重复 EOT 当作独立样本。Qt 是 clean frozen 目标敏感性代理，Rnt 是 clean NT 风险代理，均不等于已证明的因果贡献。

| 区域 | 有效图数 | Utility | 同 RMS 均匀缩放 |
|---|---:|---:|---:|
| Person 总体 | 32 | 97.0009% | 97.0009% |
| 高 Qt（Qt≥0.9） | 31 | **99.9035%** | 96.9023% |
| 高 Rnt（Rnt≥0.9） | 24 | **89.2463%** | 95.9984% |
| 高 Rnt、低 Qt（Qt≤0.1） | 22 | **74.0028%** | 95.8053% |
| Person / NT overlap | 19 | 93.2551% | 96.1719% |
| Person exclusive | 29 | 97.4682% | 97.1997% |
| Qt 加权能量 | 31 | 97.7293% | 96.9032% |
| Rnt 加权能量 | 24 | 92.4795% | 95.9994% |

在所有 22 张有“高风险、低 Qt”有效区域的图上，Utility 的该区域能量均低于均匀缩放；两 batch 方向一致。高 Qt 区域为 23 张更高、0 张更低、8 张相等；高 Rnt 区域为 23 张更低、0 张更高、1 张相等。它支持“相同总强度下将扰动更多留给高 Qt、从高 Rnt 低 Qt 区域移走”的有限结论。

两例有效但过小 Person GT 按原规则保留 W=1、排除 Qt 归因，没有删 GT；一张图没有有效 Qt 归因区域，因此高 Qt 统计为 31 而非 32。无 NT 的 8 张图保持 W=1。高保护能量统计有 37/320 个零值/并列导致的未定义项，按原规则保留 null 并排除，不能填成 0 或 1。

![同状态空间能量对照](../research_workspace/analysis/tfss_utility_static_20261017/analysis/STATIC_SPATIAL_ENERGY.png)

## 5. 实际目标与 NT 代理

目标保护项相对同状态 S0 的工程 GateB，两臂均通过。Utility 的保护 Fourier 梯度范数比中位数为 0.988055，均匀缩放为 0.982074；高保护区域算子能量中位数为 0.980134 / 0.961578。梯度 cosine 中位数为 0.999547 / 0.999959，**均匀缩放在 cosine 上略更接近 S0**，不能声称 U 每个指标都更好。保护目标相对变化均值为 −3.2858e-5 / +8.1832e-6。以上是 surrogate 目标代理，不是 Person AP。

| NT 诊断（越低越接近保护目标） | 同状态 S0 | Utility | 同 RMS 均匀缩放 |
|---|---:|---:|---:|
| 原 NT 特征漂移 L_ret 均值 | 6.2432077e-5 | **5.8339808e-5** | 5.9547485e-5 |
| 46 个真实 NT GT 的 signed loss delta 均值 | 0.00347847 | **0.00256339** | 0.00320174 |
| 12 个已覆盖 NT 类的等类均值 | 0.00850891 | **0.00711764** | 0.00817266 |

L_ret 总体比均匀缩放低 **2.0281%**，但 batch0 仅低 0.0586%，batch1 低 3.7291%，收益主要来自 batch1。真实 NT GT 诊断读取的是**第五步更新前 / EOT2**，不是五次更新后模型或 victim 结果；clean 分支对应损失最大差为 0，clean/poison 原 native TAL 分配可能变化。

不能只看均值：46 个 GT 中，U 的 signed loss delta 为 **21 个更低、24 个更高、1 个相等**；配对差中位数 +2.82526e-5，略偏向均匀缩放。24 张 NT 图按图均值为 12 张更低、11 张更高、1 张相等；12 类均值为 8 类更低、4 类更高。更高的类是 bicycle、bus、car、train；sofa 的较大下降（−0.00847936）对平均改善有明显贡献。7 个未覆盖 NT 类无证据，不能外推为 19 类均改善。

因此：空间重分配证据明确，**真实 NT 效用代理是均值正向、个体混合**。signed loss delta 是 poison−clean 的带符号诊断，不能换算为 AP、背景吸收或 Person-to-Dog 因果机制。

![实际 NT 特征漂移](../research_workspace/analysis/tfss_utility_static_20261017/analysis/STATIC_L_RET.png)

## 6. 是否值得完整 Utility 实验

**建议：值得提出一轮完整 TFSS-Utility 验证。** 理由是本次控制失败已经定位为可达性问题；同状态对照有 320/320 配对通过双门，空间重分配在两 batch 和全部 22 张关键风险区域图上方向一致；原目标保护门保持；L_ret 与真实 NT GT 平均代理均比同 RMS 均匀缩放更低。

这是一项“值得验证”的研究决策，不是已经证明有效：只有固定 TRAIN32、seed0，两个计算 batch 并非独立随机种子；均值 NT 收益较小且不均匀，逐 GT 中位数未改善，4 类均值更差；没有新 victim/AP、完整噪声或收敛证据。单靠静态空间指标不能选定最终方法。

本轮按用户六项范围结束，没有启动完整 TFSS、追加 eta/方法搜索、补做独立 Uniform 第五次优化更新或改写 S0/C/U 日程。若未来另行授权完整 U，需维持原 S0 的全程协议、标签和 clean validation、预算 8/4/0、匹配 random-init YOLOv8n200、last@200 主结果及 best 次结果，并报告完整 VOC20 target/NT/all 指标；本报告不是该启动授权。

## 7. 可核验的证据

新小证据根：[`tfss_utility_static_20261017`](../research_workspace/analysis/tfss_utility_static_20261017/)。

- [失败重建原始 JSON](../research_workspace/analysis/tfss_utility_static_20261017/measure/FAILURE_RECOVERY.json)、[CPU 优化器/张量核验](../research_workspace/analysis/tfss_utility_static_20261017/analysis/RECOVERY_CPU_METADATA.json)、[原状态对应证明](../research_workspace/analysis/tfss_utility_static_20261017/evidence/UTILITY_STATIC_RECOVERY_CPU_PROOF_20261017.json)。
- [完整测量摘要](../research_workspace/analysis/tfss_utility_static_20261017/analysis/SUMMARY.json)、[逐 GT / 分 batch 补充汇总](../research_workspace/analysis/tfss_utility_static_20261017/analysis/ADDITIONAL_PAIRED_SUMMARY.json)、[逐 GT CSV](../research_workspace/analysis/tfss_utility_static_20261017/analysis/NT_GT_PAIRED.csv)、[逐类 CSV](../research_workspace/analysis/tfss_utility_static_20261017/analysis/NT_CLASSWISE.csv)。
- [20 个原保护梯度完全重放](../research_workspace/analysis/tfss_utility_static_20261017/evidence/UTILITY_STATIC_PROTECTION_GRADIENT_REPLAY_20261017.json)、[固定协议](../research_workspace/analysis/tfss_utility_static_20261017/evidence/FROZEN_STATIC_PROTOCOL.json)、[唯一新静态审查](../research_workspace/analysis/tfss_utility_static_20261017/evidence/INDEPENDENT_REVIEW.md)。
- [最终验证](../research_workspace/analysis/tfss_utility_static_20261017/FINAL_VALIDATION.json)、[判断 JSON](../research_workspace/analysis/tfss_utility_static_20261017/DECISION.json)、[归档 SHA manifest](../research_workspace/analysis/tfss_utility_static_20261017/MANIFEST.json)。

旧独立轨迹结果仍按原文保留于 `docs/tfss/short_comparison_20261017`，没有把本轮静态结果写回旧运行。S0/S1/S2/S3 及其归档未改动。本轮新代码提交仅用于固定远程执行源；报告和小证据本地交付，没有追加 GitHub 发布。
