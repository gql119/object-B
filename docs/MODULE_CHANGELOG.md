# S2/S3 模块变更记录

原 S0/S1 源码及其正在运行的文件均未修改。父版本未选择，所有新设计仍需真实 train 准入。

| ID | 类别 | 改动及理由 | 数值/验证范围 | 原机制与可比性 |
|---|---|---|---|---|
| CNLP-001 | CNLP_DESIGN_CHANGE | 新增独立 clean/poison functional SGD 和 query damage；比较训练动态而不是复写 L_ret/KD | CPU 有限差分最大误差 1.97e-11；native YOLO J=1/3/5 二阶链有效 | 不改 Parent；两实验共享 CNLP |
| CNLP-002 | CNLP_DESIGN_CHANGE | clean-updated native TAL 共享 query assignment；positive/object/class reduction；缺失 NT supervision fail | 合成 YOLO matching/合法空类/缺匹配拒绝测试 | 真 VOC 覆盖与尺度尚未通过，未冻结方法 |
| CUP-001 | CUP_DESIGN_CHANGE | 前一步 EMA、EOT 均值后提交、signed damage clip、bounded simplex 权重 | CPU sum=1、上下界、缺席/极端值、恢复测试 | S3 唯一新差异；beta/tau/clip/上下界 UNSET |
| CODE-001 | BUG_FIX | CUP legal no-NT 分支保留原 poison 连通零值，避免 raw loss 脱离计算图 | 零梯度连通测试通过 | 不改变非空类损失 |
| CODE-002 | BUG_FIX | 独立 loop 新增完整 CNLP/CUP/query checkpoint 状态和边界检查 | 合成 continuous/resume 参数/Adam/scheduler/RNG/CSV 差异0 | 未改变 Parent 的 optimizer/schedule；真生产恢复待测 |
| CODE-003 | BUG_FIX | Windows 长路径用 extended path+Python 文件句柄保存；恢复 Adam completed-step 时还原 scheduler ordering marker | 长路径 CPU 恢复测试通过，LR/Adam 不变 | IO/恢复状态修复，不改损失/训练日程 |
| TEST-001 | BUG_FIX | 权重下界允许相等，测试改为非严格有序断言 | bounded-simplex tests通过 | 测试错误修复，无方法变更 |

当前未选择 J/lambda/CUP 参数，未完成真实 GPU 耗时/显存/类间覆盖/正式审查，未启动 S2/S3 或重新跑 S1。
