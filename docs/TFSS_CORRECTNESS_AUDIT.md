# TFSS 正确性与准入最终状态 — 2026-10-17

|核验|实际状态|
|---|---|
|历史S0原45文件/协议/参考模型|SHA保持，运行时S0 GitSHA未知|
|当前CPU组件/静态检查|10通过；原科学计算不变|
|真实S0/no-op等价|两独立batch各5步，最大差0|
|批准RMS双门|608个完成配对通过，max abs1.01638e-8 / rel2.10198e-6|
|真实smoke|v3第一批6方法1步通过；小GT修复后v4第二批6方法1步通过|
|native NT覆盖/Qt有效域|本次32图/46NTGT；12NT类；2有效小PersonGT按invalidQt保留W1/归因排除|
|独立科学审查|一次，source2abe科学正确/allow_run；小GT后sourcee962主代理真实闭合，不冒充新的独立通过|
|完整C五步子比较|两批均完成；目标保护代理门通过，效用相对Uniform-C分歧|
|完整U五步子比较|INCOMPLETE，Uniform-U batch1仅4步，step5可达域guard退出|
|比较检查点|59个Adam步数/scheduler0/RNG/有限值CPU核验；全部保留|
|Frozen/Adaptive最终哈希|11个完成trial通过；失败Uniform-U最终哈希检查NOT_RUN|
|图像浮点预算|已完成前向outside0、原8/4/0检查通过；11个最终记录派生数组CPU复查通过；不是完整PNG数据集审计|
|全TFSS噪声/PNG数据集/victim200/AP|NOT_RUN、未授权|

原v1 CUDA重放和v2旧1e-9失败原证据保留；数值双门修订有人类直接批准。Uniform-U第5步匹配目标超出控制自身最大值是新结构性对照阻塞，不能用放宽误差或重复试验掩盖。没有额外审查、模型前向来补结果或新GPU任务。

来源：[逐组件状态](tfss/short_comparison_20261017/raw/status.json)、[CPU元数据](tfss/short_comparison_20261017/analysis/CPU_METADATA_AND_PROVENANCE.json)、[独立审查](tfss/short_comparison_20261017/evidence/INDEPENDENT_REVIEW.md)、[主代理闭合](tfss/short_comparison_20261017/evidence/TFSS_SMALL_GT_MAIN_CLOSURE_20261017.json)。
