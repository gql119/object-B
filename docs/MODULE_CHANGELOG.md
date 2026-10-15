# S2/S3 模块变更记录

原 S0/S1 源码及其运行产物均未修改。2026-10-15 完整 S1 归档后已按原规则选 S0；所有新设计仍需真实 train 准入。

| ID | 类别 | 改动及理由 | 数值/验证范围 | 原机制与可比性 |
|---|---|---|---|---|
| CNLP-001 | CNLP_DESIGN_CHANGE | 新增独立 clean/poison functional SGD 和 query damage；比较训练动态而不是复写 L_ret/KD | CPU 有限差分最大误差 1.97e-11；native YOLO J=1/3/5 二阶链有效 | 不改 Parent；两实验共享 CNLP |
| CNLP-002 | CNLP_DESIGN_CHANGE | clean-updated native TAL 共享 query assignment；positive/object/class reduction；缺失 NT supervision fail | 合成 YOLO matching/合法空类/缺匹配拒绝测试 | 真 VOC 覆盖与尺度尚未通过，未冻结方法 |
| CUP-001 | CUP_DESIGN_CHANGE | 前一步 EMA、EOT 均值后提交、signed damage clip、bounded simplex 权重 | CPU sum=1、上下界、缺席/极端值、恢复测试 | S3 唯一新差异；beta/tau/clip/上下界 UNSET |
| CODE-001 | BUG_FIX | CUP legal no-NT 分支保留原 poison 连通零值，避免 raw loss 脱离计算图 | 零梯度连通测试通过 | 不改变非空类损失 |
| CODE-002 | BUG_FIX | 独立 loop 新增完整 CNLP/CUP/query checkpoint 状态和边界检查 | 合成 continuous/resume 参数/Adam/scheduler/RNG/CSV 差异0 | 未改变 Parent 的 optimizer/schedule；真生产恢复待测 |
| CODE-003 | BUG_FIX | Windows 长路径用 extended path+Python 文件句柄保存；恢复 Adam completed-step 时还原 scheduler ordering marker | 长路径 CPU 恢复测试通过，LR/Adam 不变 | IO/恢复状态修复，不改损失/训练日程 |
| TEST-001 | BUG_FIX | 权重下界允许相等，测试改为非严格有序断言 | bounded-simplex tests通过 | 测试错误修复，无方法变更 |

| CODE-004 | BUG_FIX | unit-gradient 诊断新增 EOT 向量平均后的范数/方向；原 v1 只保存各 EOT 范数，不能用于预注册平均梯度标定 | CPU 反向相消/非平行向量/非 finite 三个测试；v1 全部保留，使用新 RunID 测量 | 不改损失、样本、随机数、梯度或更新；仍无 outer step/victim |

当前未选择 J/lambda/CUP 参数，未完成正式科学审查，未启动 S2/S3 或重新跑 S1。J1 v1 已完成 38 case/76 EOT，finite/模型/Fourier不变；它缺少平均梯度记录，仅作初步诊断，不能当成最终准入标定。


## 2026-10-15 后续入口准备（未部署/未执行）

- CODE-005 / BUG_FIX：新counterfactual loop的global_params诊断元数据从误导性的csb_identity改为counterfactual_identity。S0预算与计算不变；1项synthetic完整连续/断点/完成边界恢复测试通过。首次测试未进入断言，因默认sandbox临时目录不可写；改用新的显式.local_runtime目录后通过，未删除任何旧产物。
- ENTRY-001 / CNLP_DESIGN_CHANGE：新增独立cnlp_production_admission.py，准备真实canonical Parent进程对比、lambda0、S2/S3五步与3→5续跑、8/4/0像素预算和12个历史train GT边界检查。当前仅syntax检查，真实GPU NOT_RUN。
- ENTRY-002 / CNLP_DESIGN_CHANGE：新增独立run_s2_s3_pipeline.py；两锁+单次review+当前source准入+原Parent身份绑定，顺序独立完整生成/审计/随机victim200/last&best/VOC20评价。当前仅syntax检查，没有formal配置锁，不得因入口存在就启动正式工作。
- 所有科学新入口仍需当前最终candidate真实smoke和一次独立科学审查；本轮没有审查pass或正式S2/S3结果。
