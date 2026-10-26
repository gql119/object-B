# COCO S4 端到端测试报告

## 实测边界

两个实际运行采用同一已发布0113b9b计算源及相同release acd5c930f319da3ad9a0afd9136461bb14e7406b7e8142b4bbcd8823a429580f；真实已验收Teacher、原始13630清单seed0排列首16张、nativebatch16/EOT2/一次Adam。都exit0、44次实际原生前向、0val/0官方指标/0ingest，完整保留检查点和日志；主agent另用CPU只读核验145source+7native SHA、参数/梯度数组/源与数据身份/RNG，核验器不初始化CUDA。

| 项目 | CPU2215hk | CUDA2230hk |
|---|---:|---:|
|BG→Fourier梯度|0.023283958435058594|0.022981489077210426|
|rel→Fourier梯度|0.023184968158602715|0.022854655981063843|
|实际设备|cpu|cuda:0/index0（原始前向张量直接记录）|
|每次真实原生batch|16,16|16,16|
|实际Adam更新|1|1|
|原39D/P3独立坐标|39/48|39/48|
|79NT完整|是|是|
|权重实际范围|[0.8500000238418579, 0.9364238977432251, 0.9844449758529663, 1.0, 1.0]|[0.8500000238418579, 0.936423122882843, 0.9844533205032349, 1.0, 1.0]|
|PNG预算：最大/重叠/框外|2/1/0|2/1/0|

W detached、实际加权delta严格等于base*W，Frozen与Adaptive各观察2次；每次W改变真实delta且改变其Fourier导数，具体分布和数组在归档JSON。总梯度、BG/rel/ret/orth/adaptive均有限；初始TV为0属原始零频谱，不将其捏造为非零。Teacher模型状态/原checkpoint SHA未变。原生64x3 continuation包含Adam/scheduler/Python/NumPy/Torch RNG；CUDA运行包含CUDA RNG，CPU保持无CUDA上下文。

## 无训练消费者验证

实际调用Clean200/S4_200的production dependency validator、official_eval.accepted_dependencies、两个phase config validator及final.build_report；通过6项正例、10项负例（过期SHA/Teacher初始化/batch漂移/初始化状态不符/Teacher best冒充last/旧ACCEPTED不能覆盖紧急scope）。所有fixture位于execution/emergency_consumer_graph_20261026_v1，明确SIMULATED，0模型前向/0训练，未在服务器正式目录伪造READY/ACCEPTED，也未把模拟AP当科研指标。两份Victim是fresh randomYAML训练，Teacher不充当分母。

## 尚未闭合

额外执行生产phase_export.observe_forward同一AST函数的CPU负例/CUDA正例（总2前向/0更新）状态：WAITING_RESOURCES。GPU容量门检测到其他UID的占用，未创建该作业；不放宽无冲突门，也不影响已经完成的44前向CUDA验收。后续只补此检查，不重复已通过CPU/CUDA16。

当前没有完整24epoch、正式25K导出或两份Victim200结果；验证成功不等于正式实验完成。
