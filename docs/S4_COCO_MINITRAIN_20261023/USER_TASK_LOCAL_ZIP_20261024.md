# Codex 任务：使用本地 Kaggle COCO minitrain ZIP 恢复跨数据集 S4 实验

## 一、当前情况

我已经在 Windows 电脑上下载了 COCO minitrain 25K 数据集。

数据来源：

https://www.kaggle.com/datasets/trungit/coco25k?resource=download

本地完整文件路径：

`D:\迅雷下载\archive.zip`

这个 Kaggle 数据集由 HoughNet 的 COCO minitrain 作者仓库明确列为完整 25K 数据集下载来源：

https://github.com/giddyyupp/coco-minitrain

此前服务器直接下载 COCO 图像时遇到 HTTP 502、SSL 中断等问题，但官方 train/val 标注已校验。

**现在不要再重复从网络下载 COCO minitrain 25K。优先将本地 archive.zip 上传到学校服务器，完成数据校验后继续 Experiment A。**

## 二、首先判断本地文件是否可访问

请检查你当前运行环境：

1. 当前 Codex 是否可以操作我的 Windows 本地 PowerShell。
2. 是否能直接读取 `D:\迅雷下载\archive.zip`。
3. 是否已经存在连接学校服务器的 SSH 配置或 VSCode Remote-SSH 连接。
4. Windows 到学校服务器的 SSH/SFTP 是否可用。
5. 是否有已经安装的断点续传工具。

不要假设 Windows D 盘路径在 Linux 服务器上也存在。

如果你能访问 Windows 本机，并且可以通过已有 SSH 连接访问服务器，请直接负责文件上传。

如果你只能访问远程 Linux 服务器，无法读取 Windows D 盘：

- 不要反复在服务器查找 `D:\迅雷下载\archive.zip`；
- 不要尝试使用公网代理下载代替本地文件；
- 根据已有 SSH Host 配置，为我生成一条可直接执行的 Windows PowerShell 上传命令；
- 不得编造服务器地址、用户名或端口；
- 不得输出 SSH 私钥、密码或 Token。

如果可以通过现有本地自动化工具完成上传，应优先自动完成，不必让我手动逐步操作。

## 三、上传前的本地检查

先在 Windows 上检查：

- archive.zip 是否存在；
- 文件大小；
- SHA256；
- ZIP 文件结构是否完整；
- 是否能够正常读取 ZIP 成员目录；
- 是否包含实际 JPEG 图像；
- 是否有标注文件；
- 是否存在明显损坏的 ZIP 目录结构。

不需要先把所有图片解压到 Windows C 盘。

如果需要本地临时文件，优先使用 D 盘，避免占用 C 盘空间。

输出本地：

- archive.zip 大小；
- archive.zip SHA256；
- ZIP 结构测试结果。

注意：本地 SHA256 是文件传输完整性的依据，不代表该 Kaggle ZIP 必须与其他镜像 ZIP 具有相同哈希。

## 四、传输到学校服务器

请使用已有 SSH/SFTP 连接。

服务器环境遵守既定限制：

- 仅操作我自己的目录和文件；
- 不使用 sudo；
- 不修改其他用户的数据；
- 不修改其他人的 Conda 环境；
- 静态数据优先保存在学校服务器允许的 /data 目录；
- 不覆盖 VOC20 数据；
- 不覆盖旧 S4 Poisoned 数据；
- 不覆盖已有 COCO 标注。

优先使用具有断点续传能力的传输方式。

如果现有工具支持，可以使用：

- SFTP Resume；
- WinSCP 断点续传；
- 已安装的 rsync；
- 其他经过验证的 SSH 安全传输方式。

如果只能使用 scp，请先确认连接稳定，并在传输失败时保留已有文件，不要不加检查地反复传输全部数据。

传输中断后应从合法的未完成状态恢复。

服务器接收文件先写入独立临时路径，全部传输完成且校验通过后再重命名为正式文件。

上传后重新计算服务器端 SHA256，并与 Windows 本地 SHA256 比较。

必须完全一致。

如果不一致，不得进行正式解压和训练。

## 五、服务端数据处理

完成传输后：

1. 检查服务器磁盘剩余空间。
2. 检查 ZIP 完整性。
3. 安全解压到独立的 COCO minitrain 数据目录。
4. 保留原始压缩包，至少在数据审计通过前不得删除。
5. 检查是否存在路径穿越、重复文件名及异常归档成员。
6. 不要覆盖之前已验证的官方标注文件。
7. 识别压缩包实际目录结构，不要假设内部目录一定叫 train2017。
8. 识别图像与可能包含的标签。

特别注意：

Kaggle 压缩包可能使用与标准 COCO 不同的目录结构，必须根据真实文件内容处理。

## 六、与 HoughNet 官方 25K 划分核验

官方作者仓库：

https://github.com/giddyyupp/coco-minitrain

以此前已通过校验的官方 Object Detection & Segmentation JSON 为划分依据。

必须验证：

- 恰好 25,000 个官方训练 Image ID；
- 每个官方 Image ID 都有对应的可解码图像；
- 没有缺失的官方图像；
- 没有意外的额外训练图像被加入正式划分；
- 标注覆盖 COCO80 的全部类别；
- 原始 COCO JSON category_id 与 YOLO 连续类别 ID 映射正确；
- Person 的 YOLO 类别 ID 为 0；
- 图像原始宽高与官方标注匹配；
- bbox 坐标有效；
- GT 类别与实例数符合官方 JSON；
- 无 Person 图像正常保留。

如果 ZIP 包内包含标签，可以用于交叉核对。

但本实验的正式标注以之前确认的 HoughNet 官方 25K JSON 为准。

禁止从 Kaggle 内部标签随意重建另一套类别映射或训练划分。

不能重新随机抽样 25K 图像。

如果 Kaggle 镜像与官方固定划分不一致，应报告具体缺失/多余 Image ID，优先补齐少量缺失图像；在完成一致性核验前不得启动正式训练。

## 七、验证集处理

检查服务器是否已经存在：

- 完整 COCO val2017 5K 图像；
- 原始 instances_val2017.json；
- 对应合法的 YOLO80 检测标签。

如果验证集已经完整，则直接复用。

如果缺少验证图像：

- 保持 train25K 为已准备状态；
- 单独处理 val2017 5K；
- 不要重新下载训练集；
- 不要把 minitrain 图像作为验证集；
- 不要因为验证集缺失而更改既定的评估协议。

最终必须在原始、未经扰动的完整 COCO val2017 5K 上评价。

## 八、保留原始 S4，不使用 Full

当前目标：

Experiment A — Cross-Dataset Method Generalization

父线版本严格采用：

**Original S4 — Safe-Soft05 + TFSS-Utility**

不得使用 S4-TFSS-Full。

扰动预算：

- Person 独占区域：8/255；
- Person 与非目标 GT 重叠区域：4/255；
- Person 框外：0。

保留：

- TSFL；
- Contextual Background Prototypes；
- Safe-BG；
- L_BG；
- L_rel；
- L_orth；
- L_ret；
- TFSS-Utility；
- Frozen / Adaptive；
- Fourier 共享扰动优化；
- 原有优化协议和超参数。

本实验允许为了 COCO80 重新构建特征统计及重新优化 Fourier 参数，但不允许改变核心方法公式。

## 九、恢复原定 COCO Experiment A

数据审计通过后，将现有 COCO 实验状态由 BLOCKED_DOWNLOAD 或 BLOCKED_NETWORK 更新为 DATA_READY。

继续此前已建立的 COCO80 实验流程：

1. 检查 COCO80 代码兼容性。
2. 完成 Person ID、80 类检测头和 79 个非目标子空间的适配。
3. 验证 Safe-BG 在 COCO80 下仍保持原始数学定义。
4. 验证 TFSS、图像尺寸和 GT 坐标一致。
5. 检查当前服务器是否存在正在运行的 GPU 实验。
6. 在不影响既有任务的条件下准备或运行 COCO80 Teacher。
7. 重新优化原始 S4 Fourier 扰动。
8. 导出完整 COCO minitrain 25K Poisoned 训练集。
9. 分别训练 Clean 与 S4 Poisoned YOLOv8n Victim。
10. 在干净 val2017 5K 上计算全部评价指标。

不要重新生成 VOC20 扰动。

不要新增其他扰动生成器。

不要覆盖旧实验。

如果尚未完成原始实验代码的 GitHub Snapshot，请首先核查其状态；需要归档的代码必须先安全推送到现有 GitHub 仓库，且不得上传数据集或大型权重。

## 十、与现有实验调度配合

如果 Cutout、CutMix、MixUp 已经全部完成且通过验收，则不要再次运行。

OP 如果尚未通过检测适配协议审批，继续保持独立阻塞状态。

COCO Experiment A 需要独立结果目录与任务状态。

不要抢占正在运行的 GPU，不要创建多个相互冲突的调度器。

如果既有服务器端 Pipeline 已经支持 COCO 后续阶段，优先恢复该 Pipeline，不要重建第二套相同任务。

## 十一、报告与结果

输出：

- COCO_LOCAL_ZIP_TRANSFER_REPORT.md
- COCO_MINITRAIN_IMAGE_AUDIT.md
- COCO_MINITRAIN_MANIFEST.json
- COCO_CATEGORY_MAPPING.json
- COCO_EXPERIMENT_A_STATUS.md

必须报告：

1. Windows 本地 ZIP 是否成功访问。
2. 本地与服务器端 SHA256 是否一致。
3. 服务器数据目录。
4. 实际图像数量。
5. 官方 25K Image ID 是否全部匹配。
6. COCO80 类别映射是否通过。
7. val2017 5K 是否完整。
8. 当前 COCO 实验阶段。
9. 是否已恢复后续实验队列。
10. 任何尚未解决的问题。

如果你无法操作 Windows 本地文件，请直接输出针对当前已有 SSH 配置的最简 PowerShell 上传命令和验证步骤。

不要把本地文件不存在于远程服务器，误判为数据集不可用。

## 最终要求

**当前最重要的任务是使用已经下载好的 `D:\迅雷下载\archive.zip`，完成 Windows 到学校服务器的数据转移，并严格核验它与 HoughNet 官方 COCO minitrain 25K 固定划分的一致性。**

数据验证通过后恢复 Experiment A。

原始 S4 的算法与重叠区 4/255 约束必须保持不变。
