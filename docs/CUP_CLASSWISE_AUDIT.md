# CUP CPU 正确性记录

2026-10-14：CPU_CORRECTNESS_PASSED；真实类间尺度/稀有类/权重时间轨迹 NOT_RUN。

检查覆盖：初始均匀权重；damage 历史增加后的保护优先级变化；1/2/3/19 有效类别的权重和为 1、正下界和上界；极端 damage clip；缺席历史不变；EOT 内权重不变、一步只提交一次；NaN 输入事务拒绝且状态不改变；重放状态后后续权重相同；uniform CUP 与 S2 聚合完全一致。

合成完整外层恢复比较：Fourier、CUP EMA、Adam、scheduler、Torch RNG、diagnostics CSV 差异为 0；完成边界恢复额外更新数为 0。全部使用生成的 CPU 小图和简化 outer loss，不是正式 Parent 的真实 GPU continuation。

没有真实类别有效样本数、EMA damage、权重 entropy/分布或最终 AP 数据。不存在对 Dog/Horse/Bicycle 的正式保护结论。测试夹具的参数未用于任何实验锁。
