# 本地验证、预算及导出证据

可重放命令：`python -B scripts/check_s4_tfss_full_cpu.py`。完整实际结果见 CPU_VALIDATION_REPORT.json 和 CPU_TEST_LOG.txt；原始失败开发探针不作为通过证据。

已覆盖：
- 无重叠、Person+Dog、Person+Horse、多个Person、多个NT共同重叠、无Person、小目标及边界目标。
- 实际Fourier/JND/EOT/TFSS组合：相同输入/参数/RNG下，无重叠新旧输出、raw、JND及Fourier梯度逐值一致；含重叠样本交集的pre-image-clamp delta恰为原值2倍，交集外相同，W完全相同。
- 梯度真实回到Fourier参数；Adaptive的_last_poison指向TFSS缩放后的实际图像。
- trainer/generator在相同导出阶段条件下的数值一致；旧half noise checkpoint被拒绝。
- uint8 PNG保存后重新解码：框外0，绝对差≤8。饱和投影探针在GT交集实际达到8级，说明不存在残留4级硬上限。
- 真实随机初始化YOLOv8n（本地YAML、无权重下载）在CPU上计算原生GT梯度；half/full的Q/R和valid完全相同，缓存miss→hit一致、RNG恢复、教师参数梯度为空。
- 原始45个canonical源文件SHA保持一致；独立outer loop只改变样本准入表达式，原Adam/scheduler/EOT/frequency数学AST不变。
- overlap诊断由GT几何重建；无重叠与有重叠batch的CSV schema一致，空区域记录null，不伪造为有观测的0。
- 队列准入测试：即使假定GPU空闲，只要原pipeline没有COMPLETED_AND_ACCEPTED及所有阶段ACCEPTED，仍拒绝；队列冲突、过期或身份不符的receipt同样不能正式启动。

本地磁盘不足64GiB，缓存单测仅对“磁盘剩余量返回值”使用测试fixture，以写入总量小于1MiB的小缓存。真实Q/R、模型/原生loss梯度和缓存文件均未mock。这不代表远程资源检查通过，生产容量门槛保持原值。

## 尚未执行

- 正式预训练教师的GPU Frozen/Adaptive完整outer loss、两个实际Adam更新及真实训练图像导出smoke。
- 本候选的一次独立科学实现审查（必须在当前commit GPU smoke后调用）。
- 完整16551训练图/4952 clean val导出审计；6095目标图/13256 Person GT全覆盖。
- 正式24轮/9144更新Fourier优化、victim200、AP/Recall/PUS/WCD/TCD3。
- LPIPS：项目未锁定合法本地LPIPS模型权重和实现，记录NOT_RUN，不下载或伪造；PSNR有直接数值实现，SSIM沿原有已安装实现可选记录。

“本地PNG导出检查通过”仅指实际合成输入的组件检查，不冒充完整VOC数据审计。正式导出还会逐图校验原GT/图像尺寸/labels、无Person图像字节一致、clean val字节一致、无重复缺失ID、每个Person GT改变、Manifest哈希和8/0预算。


2026-10-22 identity-binding regression: 28 CPU tests, zero failures/errors/skips. New test exercises separate final training-state and export-parameter byte identities and rejects cross-binding or changed export bytes. v1 preflight failure retained; current candidate GPU path remains unvalidated until v2 completes.
