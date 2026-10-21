# AT 独立审计（不授权运行）

旧实现：robustness_pgd_at.py，eps4/255、PGD10步、step0.8/255、uniform random start、逐步Linf projection和[0,1]clamp，最大化原YOLO box+cls+DFL损失，在native preprocess后、outer SGD前产生新batch。内循环model.eval冻结BN统计、临时冻结model参数只求input梯度，finally恢复train与requires_grad。此为旧检测适配参数，不声称REM原文同样参数。与S4既有8/255预算完全不同。

正式AT需Clean+AT与S4+AT各200epoch、batch36/workers16/AMPtrue；待A/B结束后单独实际路径验证、资源成本测量和后续人类明确启动。旧Clean AT只有121epoch未完成，不能复用为正式结果。当前AT状态NOT_RUN。

成本：每个outer batch新增10次input-gradient前反传及外循环训练，计算量可显著高于普通victim；尚无本轮实测秒/batch、GPU峰值，时间预算UNKNOWN，不把10倍计算结构粗估冒充实测10倍耗时。原S4普通victim约4小时只是参照，AT准确预算须A/B完成后的有界成本测试。

[REM官方论文](https://arxiv.org/abs/2203.14533)与[官方代码](https://github.com/fshp971/robust-unlearnable-examples)是审查来源；此文件本身不启动任何AT命令。
