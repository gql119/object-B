# Uniform-U：匹配控制可达域阻塞，不是原FP32误差门 — 2026-10-17

已批准最近可达FP32端点及absolute≤1e-7/relative≤1e-4双门，608个已完成配对均通过。失败发生在tfss_cu_short_batch1_train16_seed0_20261017_v2/sourcee962，Uniform-U第5步：`Uniform target exceeds own S0 maximum: no silent amplification`。控制器failed/exit1，workload status.json中的running是异常后陈旧字段，连接重置不是科学失败原因。

该guard先测同图、同支持域、原FP32图像裁剪后的控制RMS最大值（alpha=1）；若候选target高于此值且误差超过任一批准门，则退出。确切失败图ID、RMS上界和超出幅度未进入异常文本，记UNKNOWN；没有为打印细节而追加GPU前向/更新。59个已完成检查点均保留，CPU仅检查元数据、已有数组和图形，不替代GPU数值量化。

机制解释是由代码作出的推断：空间权重≤1只保证对候选自身相同原delta的缩放不增幅；不同方法独立优化后Fourier系数/随机历史不同，不保证候选RMS≤另一条Uniform轨迹的alpha=1 RMS。无NT图W=1同样只是算子局部恒等，不保证其共享优化后的系数等于独立S0。

已完成：batch0所有六方法5步，batch1前五方法5步、Uniform-U4步。缺失：batch1 Uniform-U第5步/最终NTGT诊断/最终扰动图。U现有匹配窗口可描述，但不能宣称完整两批优胜。C完整子比较单独保留；其代理仍分歧，不由U控制失败推断C优胜。

处置：禁止自动重试GPU/重用RunID/重跑完成组件、alpha>1、改FP64前向/amp、像素校正、静默丢弃Uniform、调整eta或AP调参。人类本次只批准数值误差修订，没有批准改变控制可达域或优化轨迹。任何新对照设计必须先明确科学问题/范围并获得人类确认。当前INSUFFICIENT_EVIDENCE，不启动正式TFSS。
