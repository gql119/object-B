# CNLP 梯度正确性记录

2026-10-14：CPU_CORRECTNESS_PASSED；真实 VOC 范数/余弦分位数、Person suppression proxy、GPU/生产 Parent 集成准入 NOT_RUN；lambda/J 未标定。

合成标量回归中，delta=-0.1；测试 inner LR=0.1，仅作为有限差分检查夹具：

| J | autograd d(damage)/d(delta) | 有限差分 | 绝对误差 |
|---|---:|---:|---:|
| 1 | -0.025893504 | -0.025893504007612478 | 7.612476621288167e-12 |
| 3 | -0.0726276981391617 | -0.0726276981402596 | 1.0978995490518173e-12 |
| 5 | -0.11333601249741423 | -0.11333601247776048 | 1.965375084900245e-11 |

合成 native YOLOv8n/VOC20 head：J=1/3/5 的共享 query TAL/GT-positive box-BCE19-DFL loss 对 support delta 的梯度均 finite/nonzero；delta=0 的 clean/poison query loss 相等，base model 状态不变。此处查询 GT 不代表真实 VOC 覆盖；不证明监督覆盖、训练收益、收敛或 victim AP 改善。

CUP 合法 no-NT 分支仍保留连通零梯度；lambda=0 等价探针保持 Parent loss 与其梯度。真实校准须按固定 train-only 集合报告原始/加权范数、方向余弦、冲突频率和分位数，不能从此表确定 lambda。
