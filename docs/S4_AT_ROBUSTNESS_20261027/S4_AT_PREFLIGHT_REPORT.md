# Original S4 AT-4/AT-8 预检报告

当前状态：CPU 预检通过；CUDA、科学审查和正式训练均为 Pending。

候选运行源码：`a8c3f4cc08b5392804e2b30148398096feaaa943`，独立分支 `codex/s4-at-robustness-20261027`。
新增内容仅扩展 Victim 侧 AT，不重新生成 S4、不使用 S4-Full、不合入 COCO 风险扩展。

## 已通过的实际检查

- 原始继承文件 60 个字节精确；本地原生 Ultralytics 8.4.150 的 12 个 SHA 与验收记录精确。
- 服务器 Original S4 manifest/完整性报告当前 SHA 精确，包含 16,551 条材料化记录。
- 两张真实 clean VOC 与对应两张 S4 poison CPU 样本 SHA 精确；没有重新生成或修改原数据。
- 8 项单元/契约检查通过；覆盖两预算、所有整数像素与随机浮点边界、GT、四组一致性和无效分母。
- Clean/Poison 均完成真实 YOLO 原生检测 PGD10、完整 box/cls/DFL、非零输入梯度和严格预算验证。
- 每个四组实际 ATTrainer 入口均完成 10 次内层前向和一次真实原生 SGD/EMA 更新；验证前向为 0。
- 四组随机初始化参数 SHA 完全相同。内层不污染已有参数梯度、模型/BN/优化器或外层 RNG。
- 生产 Trainer 的恢复入口被实际执行；受控 CPU 下一次 PGD 输入、更新、优化器与 EMA 精确，跨组恢复被拒绝。

## 数值范围与限制

CPU 真实样本采用 batch2、imgsz160、workers0、FP32。正式配置仍为 batch36、imgsz640、workers16、AMP、200epoch。
CPU 实际样本观察到类别0/12/14；全20类别输入契约有单元覆盖，完整 VOC20 正式评价尚未运行。
最初两次 CPU 探针因 FP32 区间端点的消减误差而失败，报告均保留。修复计算精度后严格预算通过，没有添加 epsilon 容差。
恢复测试是受控 CPU 下一更新精确；不宣称 workers16/prefetch 的中断轨迹位级复现。

## 尚未通过的正式门

- 当前源码真实 CUDA batch36/imgsz640 PGD10、AMP 更新、峰值显存和 CUDA 恢复尚未测。
- 由于当前 COCO 正式运行仍占用本人 GPU 与串行控制器，未启动 AT GPU smoke、独立 reviewer 或正式作业。
- 需要 CUDA evidence 后恰好一次科学审查、主 agent closure 和实际不可变 RRCTL 绑定。
- 需要正式数据 view/完整标签一致性/抽样空间8-4-0复核及新 RAM/GPU/disk/三个IOwait窗口资源门。

四组协议 RunSpec 已锁定；实际 RRCTL launch 与 READY 仍未创建。后续顺序为 Clean-AT4 → Poison-AT4 → Clean-AT8 → Poison-AT8。
当前 AT AP/P/U/S/WCD/TCD3 全部未产生，不能由 CPU loss 或 ST 结果推算。
