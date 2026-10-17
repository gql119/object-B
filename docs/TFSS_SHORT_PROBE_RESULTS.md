# TFSS 短程结果：INCOMPLETE / INSUFFICIENT_EVIDENCE — 2026-10-17

Change：人类明确批准最近可达 FP32 alpha 端点、绝对 RMS 误差≤1e-7 且相对≤1e-4。原 alpha[0,1]、预算、公式、损失、优化器、TRAIN32与EOT2不变。确定性 CUDA 控制解决真实 S0/no-op 重放；有效小GT路由按既定 invalidQt 保留强度规范修复。一个独立审查 allow_run；之后仅主代理真实当前源码闭合，没有第二次审查。

Result：已完成预定60次比较更新中的59次。batch0六方法各5步/source2abe；batch1五方法各5步/sourcee962，Uniform-U仅4步。两个总体控制器均failed，不能称整个比较完成。batch0小GT失败前已完成的30步被复用；不存在同一源码一次完整60步。详见 [逐组件状态](tfss/short_comparison_20261017/raw/status.json)。

两个batch的完整5步 S0/no-op损失、梯度、Fourier更新最大差均0。608个已完成逐图/EOT/步RMS配对最大绝对误差1.016375449081064e-8、最大相对误差2.1019758515197285e-6，全部通过已批准双门。59个更新检查点CPU元数据验证Adam步数、scheduler0、RNG和有限数值；11个完成trial的Frozen/Adaptive哈希未变，失败的第12个trial最终模型哈希检查NOT_RUN。没有转移权重/数据、没有新增模型前向来做此元数据核验。

|记录指标（两批/各5步/EOT2）|S0|TFSS-C|TFSS-U|Uniform-C|
|---|---:|---:|---:|---:|
|实际 Person RMS 均值|0.00681079|0.00618208|0.00671336|0.00618208|
|逐图配对 RMS 平均下降|—|9.1886%|1.5160%|9.1886%|
|保护梯度 cosine 中位数|1|0.997730|0.999531|0.999364|
|保护梯度范数比中位数|1|0.954692|0.988280|0.905594|
|高保护区域算子能量保留中位数|1|0.930617|0.980258|0.819490|
|保护目标相对 S0 平均变化|—|+0.001282%|-0.017990%|+0.013979%|
|平均 L_ret / NT feature drift|0.0000632976|0.0000522982|0.0000583398|0.0000480849|
|最终记录前向的 NTGT 检测损失 poison−clean 均值（46GT）|0.00226448|0.00170154|0.00256339|0.00189944|

表中样本为固定TRAIN32/65PersonGT/46NTGT，NT仅覆盖12类，有8图无NTGT。每个完整方法的保护统计汇总320次图像记录，重复步/EOT不是320个独立实验。高保护能量有37/320 undefined，按既定规则排除并保留原始值。指标来自实际优化前向，最终GT诊断/扰动图来自第5步更新前EOT2，绝不是完成5步后的独立评价。GT匹配为native allGT/有效未覆盖GT同TAL单独匹配，clean/poison匹配可变化。

Finding：C/U均通过预注册Gate B的工程代理阈值；这不保证目标Person不可学习性。U保护梯度更近S0，C的高保护区域RMS平均只下降0.1488%，相同RMS Uniform-C下降9.4892%，支持有界的空间保留机制。C与U高NT风险区域RMS平均分别下降5.6293%/5.5530%，不能称U比C更有效减少该区域干扰。

C的L_ret较S0低17.38%，但比Uniform-C高约8.76%；NTGT signed损失变化比S0低0.00056294、比Uniform-C低0.00019790。两类效用代理给出分歧，不能称C全面优于同强度控制。U的L_ret较S0低7.83%，但NTGT signed损失变化比S0高0.00029890。12个NT类中，C和U各有6类的该变化低于S0、6类高于S0；C相对Uniform-C为5低/7高。类别和样本差异保留在CSV/JSON，不挑类报告成功。

U的完整Uniform比较缺失。现有匹配窗口（batch0五步、batch1前四步）U的L_ret较Uniform-U分别低3.23224e-8、1.41297e-6；不能将其填成完整两批最终检测效用比较，也不能混合Uniform-U仅batch0的22GT与候选两批46GT均值。

构图计时：固定32图Qt总3.55993秒，额外NT风险图1.36858秒，后者约为Qt构图时间的38.44%，平均0.04277秒/图。clean maps每批预先构建后共享；该计时不能推出完整训练加速、长程开销或AP收益。两个有效小PersonGT保留标签与W=1并排除Qt归因统计。

Next：没有C/U正式200epoch候选。U控制在第二批第5步因候选RMS超过自身alpha≤1可达上限退出；原始异常未记录具体失败图ID和超出量，这两个细节UNKNOWN，不能编造数值。双门数值修订已有效，不再放宽。要补齐该控制必须先明确新的科学对照设计/范围，不能仅重试原ID、放大alpha或改变独立优化轨迹。当前结论INSUFFICIENT_EVIDENCE；完整TFSS噪声/数据集/victim/AP仍NOT_RUN且未授权。

证据：[A/B/C汇总](tfss/short_comparison_20261017/analysis/GATE_A_B_C_SUMMARY.json)、[配对与逐类分析](tfss/short_comparison_20261017/analysis/ADDITIONAL_PAIRED_DIAGNOSTICS.json)、[检查点与来源](tfss/short_comparison_20261017/analysis/CPU_METADATA_AND_PROVENANCE.json)。16张派生图包含固定6例Qt/风险/有效域及扰动图；缺失Uniform-U格明确标NOT_RUN，无源RGB导出。
