# CPU验证与未完成门槛

实测时间 2026-10-22T03:34:23.367060+00:00：31项、failure=0、error=0、skip=0；状态PASSED，Python AST检查17文件。读取实际VOC图片0，GPU steps0；所有合成fixture保留，未删除checkpoint。

覆盖Cutout的单Person、Person+Dog、Person+Horse重叠、无Person、小目标、完全遮挡、边界、空标注；20类同策略、重叠孔洞面积并集、概率0、seed稳定、非法类/坐标拒绝。原生检测MixUp多实例/空GT与像素范围；CutMix不遮挡主图、严重截断排除、无合格donor跳过；真实YOLODataset+DataLoader合成CPU batch；实际native与概率0候选img/cls/box/batch_idx数值完全相等；close_mosaic100重建后CutMix/MixUp确实p0。还有OP投影数学、signed P/U/S和零分母、队列只读等待/身份拒绝、OP不可调用等契约。

CPU_VISUAL_AUDIT.png已人工视觉核验10组前后图与GT：Cutout拒绝时图像/框不变；允许时全框保留；CutMix只新增有效来源框；MixUp三框拼接；图像为合成输入，不是S4效果示例。

首次Windows CPU测试发现np.random.randint(2**32)默认int32越界，发生于Cutout随机种子生成；已显式使用uint64修复。失败原始结果在execution/CPU_TEST_FAILURE_V1.json/.txt保留。修复后全部测试通过（随后增加冻结协议漂移拒绝测试，共31项），没有修改算法、正式训练配置或资源阈值来规避失败。

本地torch2.14.0+cpu/numpy2.5.4与远程冻结版本不同，因此这些结果不证明远程CUDA数值等价。服务器再运行同一合成CPU契约和真实独立scope/单实例/已完成job收养检查；该凭据另存CPU_SERVER_SMOKE_VERIFIED。当前新增GPU Smoke=PENDING_GPU_SMOKE；唯一独立科学审查=PENDING_GPU_SMOKE_THEN_ONE_SCIENTIFIC_REVIEW；6次正式训练NOT_RUN；OP=BLOCKED_PROTOCOL_APPROVAL；无AP可填。

运行：`python -B -m robustness_s4_extended.validate_cpu`。测试结果JSON含确切版本、配置SHA和原始文本SHA。未运行全仓回归、性能benchmark、全量VOC遍历、模型预训练或OP方向估计。
