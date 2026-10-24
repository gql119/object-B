# Original S4 方法一致性（当前实现状态）

科学父线是generation `b1e29510e7b1ed5fa0fe361a27132d8790e65c14`；独立Victim组件是`ab626917f20917e58cf30033ac99df00938cb561`。完整既有对照见[S4_COCO_METHOD_PARITY.md](S4_COCO_METHOD_PARITY.md)和原始8数学源/2Victim组件SHA锁。新增Teacher CPU guard探针重新核对这些源均未改变；没有使用Full/S5/CUP。

8/4/0预算、STE2.01/255、TFSS-U eta.85、StrictP5/24D冻结门、SafeBG top3/rank8/alpha.5及范数恢复、P3 orth/ret、Frozen/Adaptive单步、原损失权重、Fourier24/Adam/EOT2/seed0/JND/TV不变。所有COCO teacher/统计/原型K8/79NT子空间/缓存/噪声重新生成，不能复用VOC教师/通道/最终噪声。COCO步骤数由原始>10支持门的有效Person图像数确定，不能硬套VOC9144；真实Strict空间/类秩不兼容时阻塞，不放宽阈值。

现有45 CPU契约已接受；新增train-only Teacher组件的两个定向CPU探针通过，详见TEACHER_TRAIN_ONLY_CPU_REPORT.md。新组件仅排除Teacher的val数据和调用并保留原生checkpoint；原生SGD/预处理及S4计算源不改。实际CUDA、完整loss/delta/Fourier链、新阶段独立科学审查、全串行生产入口仍未完成，不把组件CPU通过当完整方法通过。Teacher固定last200后的val仅用于独立只读报告，不参与训练、统计或选择。
