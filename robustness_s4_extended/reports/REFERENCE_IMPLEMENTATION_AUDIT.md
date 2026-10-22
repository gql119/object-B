# 原论文、官方实现、检测适配的区分

论文：[MI-UE ICLR2026, §6.3 / Table5](https://arxiv.org/html/2603.03725v1#S6.SS3)，分别测试三种增强与OP；表中是CIFAR10分类准确率，不能移作VOC检测的AP或推荐最优参数。本项目不复现MI-UE毒化方法，只审计其防御参考。

MI-UE官方代码固定 commit `4cd387978c08b0c85eed9afc0b849d0b8777691c`，四份指定源文件和requirements原始字节已留在 execution/references。每文件永久链接和SHA见 REFERENCE_SOURCE_MANIFEST.json。

| 方法 | 官方分类实现 | 已安装Ultralytics实际行为 | 本项目检测协议 |
|---|---|---|---|
| Cutout | cutout.py默认n_holes1/length16；32x32上名义面积25%，中心均匀、边界裁剪、乘零mask。dataloaders.py的CIFAR评估分支实际调用默认构造（注释提到CIFAR100长度8但调用没有传8），无单独概率分支，等价每图尝试p1；不改单标签。RandomCrop32,padding4和Flip.5仍保留 | 无直接采用的Detect Cutout入口 | 在线 uint8图，n1、名义面积.25（宽高各.5）、p1、fill0。所有GT遮挡上限.5；任何类别超限拒绝整个候选，最多8次，之后跳过；不裁GT、不删除GT。真实面积因边界裁剪及拒绝而变小，不能声称等同CIFAR强度 |
| CutMix | evaluation_sl.py按epoch创建Kornia RandomCutMixV2(input,class)，分类label mixing经loss_mix使用；requirements固定Kornia.6.10；默认num_mix1/p1/same_on_batch=False，cut_size与beta传None，该版实际generator解析为cut_size[0,1]、Beta(1,1) | Detect CutMix实际存在：Beta1，3候选区；仅接受与所有主图GT零相交区域；来源目标bbox可见面积>=.1（Detect），裁剪到粘贴区，拼接整数cls/Instances；不存在分类CE混合 | 调用该native实现，唯一额外标注保护是donor可见面积>=.5且裁剪后边长>=2px。主图GT不遮挡。p1，Beta1，3候选。不满足则跳过。严重截断目标排除；允许的截断按原生规则裁框，和Cutout不裁框的策略不同 |
| MixUp | Kornia.6.10 RandomMixUpV2(input,class)，默认p1/same_on_batch=False，lambda_val=None在该版generator对应Uniform(0,1)；loss_mix按混合label计算分类损失 | Native Detect MixUp取Beta(32,32)，uint8加权图像，拼接双方所有Instances和class IDs，不使用分类lambda*CE或额外实例损失权重 | 原生Beta32/32、p1；保留整数GT实例，双方同一arm采样。分布与Kornia分类版本不同，明确是YOLO检测适配 |

所有新增方法互斥。原生Mosaic、HSV、flip、affine、CopyPaste0不叠加独立第二套。native close_mosaic100同时把MixUp/CutMix概率置零，因此它们仅前100epoch有效；Cutout在最终Format之前、HSV/flip之后执行全部200epoch。Clean/S4均采用同一调度；这不是强制200epoch一直MixUp的另一协议。

随机机制：保留原生DataLoader worker seed和基础增强RNG；新增操作以SHA256(seed0, worker读取到的epoch, image stem, operation)生成局部seed，保存/恢复Python/NumPy全局状态。记录每worker前4次增强的GT遮挡/样本seed。预取边界可能读到旧epoch；不声称任意worker调度下Clean/S4每个batch遮挡完全一致，不声称精确DataLoader续训。相同sample/epoch/op的新增随机机制可复现；基础图像内容/原生albumentations仍可能影响数据流。

Cutout的“可见”是bbox遮挡面积代理，不能从bbox保证实例分割级可见性。实际遮挡率与跳过率可能依赖GT几何，需结合配对Clean绝对AP和完整非目标指标解释。

OP官方仓库为NeurIPS2023 What Can We Learn from Unlearnable Datasets?，固定 commit `866d1dab6420bb1ea99796c8a263f3b0562e23e9`。Step1将固定尺寸未归一化图像展平，训练无bias线性分类器CE，默认20epochs/SGD lr.1/batch1024/MultiStep50%及75%；类方向是fc.weight各行。代码中可视化部分硬编码10类3x32x32，不能直接套VOC20。Step2从同一poison identifier加载该线性模型，取权重行，normalize并QR，project_data依次扣除正交向量分量，实质X-(XQ)Q^T；reshape成图像后仍有原分类强增强与Normalize（默认未关闭），clean test默认不投影，另有project-test选项。原始project_data不把值自动夹回[0,1]；不要把本项目候选重建clip声称为原版实现。OP不是S4的语义L_orth。
