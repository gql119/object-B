# S4 鲁棒性冻结协议

用户已授权A/B完整训练。唯一Poison是原始S4固定数据，原S0只作历史源码/既有Clean权重来源，不用于任何本轮Poison训练或Smoke。无扰动优化/参数搜索/CUP路径。A顺序JPEG50→JPEG30→Gray→BDR；每项S4训练200后配对Clean200，或已核验Clean直接复用。B顺序Gauss→MeanF→MedianF→DataAug，四组Clean只重评不重训。一次只允许一个正式训练；每次新RunID/输出不覆盖。

全部seed0/randomYOLOv8n/epochs200/batch36/workers16/AMPtrue/SGD/cos_lr/close_mosaic100/cachetrue/lr.01/lrf.01/momentum.937/wd.0005，其他原生默认值及依赖须与已核验S4一致；patience0确保完整200。使用现有S4VictimTrainer，optimizer_step完全继承原生GradScaler；被动记录合法skip/backoff，16连续skip安全界限不变；持续非有限loss/model必须停止保留证据。

每组始终在原始4952cleanval评价，last@200primary/bestsecondary；全VOC20 AP50/AP75/AP50-95/R/P，Person/NT19/all。P=1-AP_t_poison/同防御AP_t_clean，U=NT_poison/同防御NT_clean，S=2PU/(P+U)；输入小数，零分母null/undefined，负P/负损伤和U>1不裁切。WCD/TCD3参照同防御Clean并保留符号。不得用跨防御Clean或舍入数计算。另报告相对原始无防御S4的绝对AP/Recall变化及相对同防御Clean的效用缺口。

资源继续既有3窗口规则，loadavg仅记录，所有窗口CPU>=95%且余量<=5%或IO>=20%才认持续争用；RAM>=32GiB/GPU空闲free>=24000MiB/util<=5%/disk>=150GiB，监测Swap/D-state/他人进程。训练中自身GPU占用合法，不能套用空闲准入误停。无权干预他人。新输出保留每轮checkpoint/full状态；没有worker/prefetch RNG证明时不伪装精确续训。

Smoke上限100 actual training batches，固定TRAIN32，native_S4/no-defense pair各5，四A防御Clean/S4各10，独立synthetic overflow最多10；正式指标/结果ingest关闭。当前源码通过CPU→真实GPU smoke→恰好一次独立科学实现审查后才正式训练。

AMP硬件自检复用历史防御local_amp_check：当前随机模型eval、固定zero1×3×256×256、FP32/autocast输出形状/有限性/allclose atol0.5，恢复train状态，不下载YOLO26n。该自检不执行优化更新；正式SGD/GradScaler继承原生。与S4离线跳过外部权重自检相比只增加显式自检证据，实际AMP必须为True，否则立即阻断；基线配对共享同一自检。
