# CUP-S4 变更记录

Change: 创建独立 S5 worktree/branch；归档精确用户任务与 S4/S3 源码来源；添加仅 CPU 合成输入的 source-derived retention 审计脚本与报告。

Result: 五组 CPU FP32 用例通过；最大 scalar 分解误差 3.637978807091713e-12、gradient max difference 0、基准残差形式 no-op difference 0。45 个原 S0 文件、S4 maps/formula/损失/Victim AMP 代码与 Parent 均未修改。没有 GPU、模型、数据前向，没有优化器更新。

Finding: 原 L_ret 可按有效 P3 GT ROI 实例数分解；P5 类漂移诊断和 uniform class mean 不是等价替代。S3 权重状态机可独立复用，但原 damage/aggregate 绑定 CNLP。拟改用 P3 feature drift 作为 history 信号会改变 damage 定义，状态为 PROPOSED，尚未实施。

Next: 根据用户附件最后一节，确认具体 damage 定义适配。随后才做真实固定训练开发诊断和受影响的 production checks。S5 配置、checkpoint、AMP状态、训练日志、正式结果均 NOT_RUN；Parent、teacher、dataset、seed、source hash 与授权 SHA 见 `S5_SOURCE_AUDIT_PROVENANCE.json`。
