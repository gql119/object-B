# 最终建议与15问

**决策：RECOMMEND_TSFL_ONLY。** 仅推荐B作为下一阶段待批准的最简研究候选，不表示已证明优于S0，不是启动正式实验的授权。本轮诊断完成后停止；S3按原方案继续，TFSS等待用户确认。

## 逐项回答

1. **S0有效扰动在哪里？** MEASURED：仅Person union，重叠a0=.5、独占1；历史PNG框内RMS均值0.026253。中心/边缘能量面积密度0.9664/1.0179，近乎均匀，不能将中心默认为主要保护区域。扰动幅度与Q相关-0.0530，也不能仅凭幅度判保护价值。
2. **TSFL输入归因在哪里？** MEASURED：65对象平均自身框内质量31.12%，图合并Person union为43.95%；大量感受野延伸至周边。中心/边缘密度1.0596/0.9719，没有强中心先验。这里只定位对选中特征能量敏感的像素。
3. **与原保护梯度一致吗？** MIXED：Spearman0.3436，Top20重合36.00%，有中等信号、非强一致，30有效图；逐对象尺度结果和63有效对象保留。两张零保护图未伪造一致性。
4. **低预算覆盖高保护位置吗？** MEASURED：高保护Top20位置54.79%预算低于S0；这些位置平均少0.5270 byte，其他位置少0.1218。独占区域比例71.52%，重叠45.06%。这比“总平均强度变低”更具体，但仍是固定S0状态的代理。
5. **hard clip阻断多少？** MEASURED：30有效图、图内EOT平均后的g_out标量通道能量阻断57.67%；局部全部RGB归零像素率36.80%（32图）。相对a0*g_out的梯度范数保留68.69%、方向余弦0.5997。两种分母不同。无NT的8图走正确legacy分支，未硬裁剪。
6. **发生在高TSFL位置吗？** MEASURED：高TSFL区域g_out能量阻断58.41%，高保护区域58.00%，存在重合；不能据此宣称全部被挡的梯度都是有效UE收益。
7. **中心渐弱会伤边缘关键位置吗？** MEASURED/HYPOTHESIS：它人为衰减边缘，而观测保护能量没有明显中心富集。A在高保护区保留实际能量48.83%，B60.42%；所以中心先验有不必要损失的风险。没有像素语义/肢体因果标注，不断言所有边缘部位都是关键特征。
8. **TSFL-only优于中心的梯度保留吗？** MEASURED：局部固定g_out范数79.00% vs69.56%，实际RMS69.84% vs68.93%原S0，强度相近。B也优于请求能量匹配UniformB的69.64%梯度保留；其优势不是单纯比A留更多总能量，但控制仍有最大1.869%有效能量偏差。
9. **Fusion是否互补？** UNVERIFIED：C保留81.23%梯度、高保护区65.04%能量，但实际RMS76.48%，明显强于B69.84%，嵌套eta最低0.625。不能把较强幅度当互补证明；增加sigma/rho暂缺必要证据。
10. **平均实际强度有差异吗？** MEASURED：A/B/C实际RMS为S0的68.93%/69.84%/76.48%，实际MAD为68.03%/68.35%/76.21%；均值权重0.6805/0.6854/0.7628。比例按每图计算后平均，不能与均值之比混用。
11. **只是全局减噪吗？** MIXED：三者都有明显整体减弱。B与UniformB实际RMS近同，但高保护能量60.42% vs48.69%，说明还存在分配信号。反面是Uniform改善NT漂移更多；B未证明得到全面更好的权衡。
12. **S1退化更支持哪种机制？** HYPOTHESIS：预算布局和hard clip均有直接代理证据。固定预算对照和common-g_out VJP支持算子改变方向；高保护位置预算下降支持位置冲突。Fourier范数比仍达0.9834、余弦0.8363，说明总norm不能排除局部截断。2.01floor/2cap仅涉及0.92%像素、1.64%保护能量，不能当主因。没有测S1历史每一步，不能分摊最终+6.2677pp AP的因果比例。
13. **值得设计TFSS正式实验吗？** PROPOSED：值得进一步设计一个B候选，用来证伪/验证上述分配信号；当前不承诺效果，也不跳过未来独立初始化/全程预算/非目标保持和完整victim评估的门槛。静态relation代理均劣于S0，NT代理没有一致胜过Uniform，必须如实带入下一阶段风险。
14. **最简构造是什么？** PROPOSED：B，W=eta_TFSS+(1-eta_TFSS)Q，delta=delta_base*a0*W；保持原损失、a0和上游路径。诊断eta=.5为预注册值，不是validation最优值。不要先加入中心或Fusion，不去改正在运行的S3。未来正式配置尚未批准/启动。
15. **新增参数和成本？** CODE_VERIFIED/UNVERIFIED：B增加一个eta_TFSS标量设计参数，另需冻结归因定义、3×3平滑与Q10/Q90规则；没有新增可学习网络参数。逐图一次Frozen前向并保留图，每Person一次能量反传（本32图为32前向/65对象反传），再有max合并与像素乘法。可考虑冻结clean-map复用，但本轮没建全量缓存。A需eta、sigma且无需Teacher归因；C还加rho。正式GPU时间、内存和缓存成本UNVERIFIED；本轮CPU570.12秒包含6模式/EOT及完整高阶损失，不能拿来当正式增量耗时预测。

## 候选比较与反证

|方案|框内均值权重|实际RMS/S0|实际MAD/S0|高保护区实际能量/S0|固定g_out局部梯度范数/S0|
|---|---:|---:|---:|---:|---:|
|Center-only|0.6805|68.93%|68.03%|48.83%|69.56%|
|TSFL-only|0.6854|69.84%|68.35%|60.42%|79.00%|
|Fusion|0.7628|76.48%|76.21%|65.04%|81.23%|
|UniformB|0.6974|69.89%|69.96%|48.69%|69.64%|

H1=MIXED，H2在限定配对条件下SUPPORTED，H3=MIXED。推荐B是因为它用更少先验在相近能量下保留更多原梯度，**不是**因为代理损失或正式AP证明更好。当前B的NT feature drift改善弱于Uniform，Q与原G_NT相关0.3381；“TSFL选中特异通道”不等于“其输入高响应像素对NT无影响”。这正是未来方案仍须保护NT的风险来源。

所有结论限固定分层train32、一个最终S0状态/seed、CPU和静态反事实；没有五步收敛/UE/背景吸收/Person-to-Dog证据。原S0历史Adam/RNG不全；频谱保真尚未测。推荐不自动触发任何正式方法、训练、materialization、victim或GitHub发布。

## 可追溯身份与口径

- 诊断源：`5bcae7233e03ebc6d2533f0ae311f67a68676899`；RunID `tsfl_spatial_cpu_fixed32_seed0_20261016_v1`。完整原始证据在 `raw/`；聚合见 `AGGREGATE.csv` 和 `AGGREGATION_POLICY.json`。
- Frozen/Adaptive Reference SHA：`8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`；S0 final Fourier SHA：`2490dd1ef1b95d1a1df9b8f32ce69555ec2516f55eac832ef6ff40003c1f2a77`。S0 当时 runtime Git SHA 未知；不把后来源码归档当作当时提交。
- 数据：VOC_0712_Kaggle_Ready; original clean train manifest f38fa031946d9eaec26665b09fcfebad09c2c1e9e20deff3b5770192eb5cef8f；固定训练32张/65PersonGT，seed0。完整 Image ID/GT/图像与标签SHA见 `FIXED_TRAIN_SAMPLE_MANIFEST.json`，committed-LF SHA `d15881f71f8c62ad0624f1a9c8153c0a146511978631df2017ec323373ad3522`。没有使用 validation 选样、选参数或优化。
- 特征：原始 P5/model.21，24通道 `[0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125, 128, 133, 140, 167, 173, 184, 220, 229, 237, 238, 239, 240]`；原通道选择工件未重建。归因目标为每个 Person GT 的选定通道**平方响应均值**，使用原 fractional feature mask；RGB输入梯度L2。先3×3平滑，再各GT框内Q10/Q90归一化，多Person取max。
- 原图坐标；batch内右/下pad至32倍数，沿原实现letterbox至640，再做原feature mask映射；统计图裁回原尺寸。细节与源码身份见 `SOURCE_AUDIT.md` / `SOURCE_METADATA.json` / `PROVENANCE.json`。
- 实际设备 CPU FP32、单线程、nice19、CUDA隐藏；耗时 `570.122s`。batch16/EOT2，模型及Fourier参数未变，optimizer updates0。真实原S0 Frozen损失和Adaptive高阶路径均执行；不是普通分类梯度。`g_protect=||mean_EOT(g_out)||_RGB`；局部Jacobian能量逐EOT计算。
- EOT先在同一图内平均，再逐图等权；对象表按GT计数；共享Fourier余弦按2个batch计数。预算另给pixel-weighted结果；二者不能混用。32张为分层开发样本，不代表总体分布，不作显著性或因果检验。
- 单图/修复probe及失败统计保留。唯一独立审查原结论为 scientifically_incorrect；B1–B3及主代理发现的B4已由当前源真实CPU probe闭环，**没有第二次审查或虚构独立通过**。见 `execution/lean_review/MAIN_AGENT_CLOSURE.json`。
