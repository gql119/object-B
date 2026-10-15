# S2/S3 代码准备

## Change

用户 2026-10-14 “先写相关代码”授权在 S1 尚未完成时准备独立本地实现。建立 `codex/safe-soft05-s2-s3-counterfactual-20261014`；独立 CNLP 双路径、共享 query assignment、CUP 历史聚合、outer loop/checkpoint、train-query manifest 和父版本准入代码。S1、S0 历史源文件均未更改。

## Result

55 项 CPU 测试通过，47 新测试和8历史契约；native YOLOv8n合成输入 J=1/3/5 的虚拟更新梯度链可达，有限差分与 autograd 最大误差1.97e-11；synthetic outer-loop continuation 的参数/Adam/scheduler/RNG/CUP/CSV一致。源码状态为本地未提交修改；详见 docs/S2_S3_CPU_VALIDATION.json。

## Finding

SUPPORTED（仅正确性）：当前 synthetic CPU 范围的梯度链、随机状态/BN隔离、历史聚合和checkpoint恢复。OPEN：真实 NT GT覆盖/损失尺度、稀有类稳定性、CNLP权重/方向、GPU显存耗时、生产 Parent 集成及最终检测效用。CPU 通过不等于科研有效、收敛或正式准入。新任务母版 SHA182a145937a888d489eaf232ca7229a0c252c46626f4cec65d2aea4581b50311。

## Next

等待 S1 完整归档，按严格1pp双条件锁 Parent，绑定真实预算/source/config/reference身份。仅用 train开发数据完成CNLP/CUP审计和标定，锁定两实验协议，再独立顺序S2→S3。Parent、J、lambda_CNLP、CUP参数未选；真实VOC/GPU准入/正式noise/victim均NOT_RUN；不得上传未完成实验的代码至GitHub。
