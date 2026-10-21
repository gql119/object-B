# 单变量方法差异与源码审计

正式对照是已核验的 S4 — Safe-Soft05 + TFSS-Utility，主评价 last.pt@200。原始报告位于 `docs/S4_AMP_RECOVERY_20261018/S4_COMPARISON.json`，本任务锁定其文件 SHA、噪声 SHA、数据 Manifest SHA 和 last.pt SHA。历史精确值为 Person AP50=0.028158895413428475、NT mAP50=0.6937870442308225；S 从同一 Clean 分母重新计算，展示约0.935330。没有使用鲁棒性预处理数据充当原始 S4。

## 唯一计算变量

原始 support = Person union × (1 - 0.5 × NT union)；新 support = binary Person union。两者的非零几何域相同，多个 Person 不叠加强度。

保持原链：相同 Fourier 场 → 相同 JND/训练 EOT → 相同 clamp/STE最小幅度 → 最终 support缩放 → 原有 detached TFSS-U → 图像 clamp。Full 的重叠区基础上限是8/255，TFSS仍可把强度降低至其原有范围；“统一8/255预算”不表示每个像素必须达到8/255。

## 审计位置及处理

| 位置 | 原行为/风险 | 本次处理 |
|---|---|---|
| legacy_kproto_ret.py / _bbox_half_support | 构造1/.5/0；旧模式才启用量化前floor | 原文件字节不变；FullTargetSupport显式新增模式，并复用实际 _TAUSBCommon Fourier/JND/EOT 实现 |
| _StrictTargetSupport._compose_delta_batched | 2.01/255 STE floor 后乘support | Full保留完全相同floor、符号边界规则和执行顺序，仅乘二值support；不把floor顺便降为1级 |
| tausb_universal.py / generator | 模式名控制最小support阈值，普通模式会跳过≤10像素目标 | Full独立generator静态保留原生成函数，仅保持bbox-half原有的 >0 导出门槛 |
| s4_optimization.py | support.sum()>10把幅度与样本准入耦合 | 独立optimizer保留原样本准入数值，单独按原half加权面积判断；该判断不进入delta、loss、budget或导出。其余优化器函数AST一致 |
| tfss_spatial.py | support只校验{0,.5,1}，W只根据support>0限制几何域 | {0,1}合法且同域；公式、eta=.85、detached行为和Adaptive实际poison输入保持原实现 |
| tfss_clean_maps.py / tfss_nt_sensitivity.py | Q/R使用clean图像、GT和support>0；不消费.5幅度 | 原文件字节不变；CPU真实随机YOLO原生梯度验证Q/R相同。正式教师GPU仍待验证。选择新目录和新身份，不复用旧缓存 |
| s4_tfss.py | 浮点预算按传入support×eps检查；旧overlap统计依赖support==.5 | 原文件不改；新trainer保持loss继承，补齐独立GT几何交集诊断，并校正旧overlap_linf字段 |
| 新生产导出 | 原脚本独立实例化旧generator，并有旧预算文案 | 独立Full入口显式实例化Full generator，逐图PNG投影/重读审计采用8/0，要求新noise的source/config/support身份 |
| checkpoint/cache/Manifest | 错用旧noise或旧缓存会污染对照 | support_mode、experiment_id、source/config/cache身份写入；旧half checkpoint构造时拒绝 |

训练样本准入特别保留原规则，是为了避免一张“加权面积≤10但二值面积>10”的小重叠样本随support改变而被额外加入outer batch。导出仍覆盖所有有效Person GT；本次没有给重叠像素保留4/255上限。实际数据上的原始支持图像数、样本数和GT数在远程source audit再核验。

ROI、上下文排除、TSFL通道顺序、Safe-BG、worst-3正交、Frozen/Adaptive、各loss及权重均继承原S4。非重叠样本的组合输出、raw/JND、Fourier梯度已数值比较，而非只据源码推断等价。

## 对结果的解释边界

这一消融隔离“取消固定减半策略”的总体影响。重新优化的是共享Fourier参数，因此最终非重叠区域也可能发生间接变化。区域RMS、重叠强度和AP同时变化，不能单独证明最终性能主要由重叠像素介导。报告会给出相同GT几何下的原S4/Full区域强度对照；不会将此表述为因果中介证明。

Person AP下降而NT/WCD恶化，只支持更强目标保护伴随更大附带损伤，不支持TFSS已充分取代重叠保护。单seed不做显著性宣称，不据smoke或loss预测正式AP。
