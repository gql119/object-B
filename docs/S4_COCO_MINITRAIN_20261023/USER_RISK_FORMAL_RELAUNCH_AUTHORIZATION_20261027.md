# Direct human authorization: independent COCO80 risk protocol formal recovery

当前 COCO80 TFSS 风险补充分支已经完成完整 25K 统计、CPU/CUDA 验证和 VOC 回归测试，修复质量已达到**可以考虑恢复正式实验**的程度。

我授权你在满足以下条件后，自主恢复正式实验，无须再次询问：

1. **独立协议：** 使用 `coco80-ntgt-native-tal-then-gt-risk-v1`，明确记录新增的 (R_nt) 边界定义，不得将其宣称为与严格 Original S4 完全等价。
2. **正式预检：** 核验 Git Commit、配置、Teacher、校准缓存、39D 特征、全部 GT、TFSS 训练接线、Fourier 梯度、8/4/0 扰动预算及服务器 GPU 状态。
3. **复用有效成果：** 保留并复用已验收的 Teacher、25K 校准、背景原型和非目标语义子空间，不重复有效工作。
4. **重新优化：** 创建独立 RunID，从原始 Fourier 初始化重新执行，不得从此前错误的 12 步 Checkpoint 续训。
5. **自动完成后续实验：** 预检全部通过后，依次执行 Fourier 优化 → Poisoned 25K 导出 → Clean/Poisoned Victim 200 epoch → 干净 val2017 5K 评价。
6. **科学一致性：** 保持 Original S4 其他核心方法、损失函数和超参数不变，不使用 S4-Full。保留完整日志与 GitHub 版本记录，不干扰其他任务。

**如果正式预检全部通过，直接恢复实验；如果发现新的方法学不一致或数据正确性问题，停止受影响阶段并报告，不得通过跳过 GT、关闭检查或修改核心算法来强行运行。**

请使用现有唯一串行控制器执行，不创建重复调度器，并报告新的 RunID、协议版本及正式预检结果。
