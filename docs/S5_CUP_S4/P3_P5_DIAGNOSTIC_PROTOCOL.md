# Stage B 固定训练开发层对照

最新授权为 `USER_TASK_LAYER_SELECTION_SOURCE.md`，允许 P3/P5 全通道对照、自主选层、按选定层 stopgrad 类漂移适配 CUP；此前的 damage-definition 等待已经解除。Stage C 前不更改正式 S5 保持层，也不运行完整扰动生成。

本次诊断配置 `configs/s5_layer_probe.json`。原 TRAIN32 manifest SHA256 `3becd3c5c9338dfb887da6163df8063a8ed44303dbacc4351f15e5ba376be5a1`；split=train、seed0、前16/后16两个独立 batch。已有分层覆盖 Person-only8、Person+Dog3、Horse5、Bicycle4、高重叠18、small Person4 等；记录所有实际非目标类与尺度，不假设 19 类都会在32张图中出现。

每个 batch 从原始 seed0/Fourier/Frozen/Adaptive 初始化运行三路：真实 S4 reference、P3 实例类分解基准、P5 全通道基准；每路5个 Adam 更新、原EOT2，总30更新。每路5更新后另做2次共同 seed0 EOT观测，无第6个更新。P3/P5均不启用CUP，不修改原 lambda_ret=.25。

S5 新 core 静态复制 `SemanticFrozenTrainer._batch_loss`，仅加两个 ROI 观察点及一个 Frozen retention reduction sink。父文件和45个原始源文件字节不动。AST 回归验证移除这三个声明变化后与原方法完全相同。MRO 保留实际 S4 空间控制、原 B2Semantic/Adaptive wrappers。原始 S4 reference 直接执行原 S4Trainer 和 optimize_s4，不实例化新 core。

在相同当前扰动状态计算两层 raw/per-instance/per-channel drift、有效GT及大小、Fourier 梯度范数/保护余弦/元素冲突/保护反向分量。不同层 raw 值不能直接排序。基准等价检查覆盖原始 loss、Fourier gradient、实际 postclamp delta、support、W以及5步后的 Fourier/frequency/Adam/scheduler/RNG；容差1e-7。

两层均保持原 GT语义和P3图像资格。P5 mask mass<1 的实例显式记 invalid，并与原P3有效GT列表比较；不以常数、忽略记录或替换ROI断言通过。P5若缺失原实例，将作为选层的不利证据，不能宣称完整GT集合保真。共同有效GT与各自全部有效集合均在原始记录中可区分。

GT检测代理复用原 S3 的 native TAL/GT reduction 代码，但输入来自同一个 Frozen 对同一实际EOT clean/poison的原始head输出；不运行任何CNLP/虚拟更新。共享detached clean原生all-GT分配，未覆盖有效GT单独在同一TAL匹配，box/BCE19/DFL和gains逐项保存。它只用于诊断，未加入优化目标。

预先选择规则：真实基准等价、finite/预算/模型不变通过后，P5还需不丢失原P3有效GT；两独立batch保护代理均不劣、梯度冲突不加重、GT检测代理与类别/尺度支持一致更好、静态和5step方向一致。弱证据或相互矛盾则选P3。完整规则作为配置归档，不按validation AP更改。未做第三组 NT-specific channel subspace 探索。

保留所有checkpoint、失败和远端派生梯度/扰动文件；仅拉小型JSON/日志，绝不拉取权重/原始RGB/数据集。计时包括观察器开销，不能冒充无观察器训练吞吐。

本诊断不是 CUP 正式候选的独立科学审查通过证据。CUP实现后仍须当前提交production GPU smoke及一次独立S5科学审查，然后完整准入才执行正式任务。
