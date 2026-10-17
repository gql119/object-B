# 2026-10-15 当前候选设计（以下旧草案作为历史）

Parent=S0，J=1，innerLR=.0001继承原Adaptive，lambda_CNLP=0.058485507823134296；support/query各1，query从完整train manifest选择且排除整个16图Parent support。显式matching_policy=native_shared_gt_coverage_v2；覆盖native GT不变，仅合法无positive GT在同一clean-updated TAL中单GT恢复，两路径共享；仍缺证据就fail。不同GT可能共用anchor，这是GT-conditioned query设计，不是整图原生loss。不是S1预算，S0固定8/4/0且原Lret等损失不动。

lambda按原固定训练开发规则计算，tail25%上限决定最终值，非零case加权/Parent梯度中位数仅0.66854%，不能描述成普遍10%。CUP提案与S2共用同一CNLP/lambda/J；只改聚合。严格模式原J1/3/5均通过，当前覆盖模式38case GPU重放记录数值差0（norm/cos/statistic，非完整GPU向量比较）；CPU完整向量差0。最终正式配置锁/一次独立科学审查仍待完成。没有最终AP或收敛结论。

证据：docs/cnlp_calibration/；execution/cnlp_coverage_v3_replay_verification.json；BASE/cnlp_small_20261015/cnlp_admission_artifacts_seed0_20261015_v1/。旧损失推导见下文。

---

# CNLP 实现草案

状态：DRAFT；真实训练开发审计和标定 NOT_RUN。原任务附件 SHA256：182a145937a888d489eaf232ca7229a0c252c46626f4cec65d2aea4581b50311。

从原 Parent Adaptive 的同一个 theta0 和全零 SGD optimizer 状态开始，两个 functional state 各执行 J 次 all-GT native support loss 更新。momentum=0、weight decay=0 与原虚拟 SGD 保持一致，inner_lr 强制继承 Parent。J 从 1/3/5 中经训练开发审计确定，目前未选择。支持输入使用 Parent 既有加噪输出及相同 clean 图像/GT/padding/letterbox，不新增随机增强；模型随机 draw 在两条路径重放。BN running statistics 不更新，affine 参数仍参与虚拟更新。

同一 clean TRAIN query 的全部身份与整个 support batch 不重叠。Clean-updated 模型生成共享 detached native all-GT TAL assignment，poison 路径使用相同 anchor、GT box 和 soft target。对每个 NT GT 的 positive locations：native soft score 归一化加权 CIoU/BCE19/DFL；BCE 输入是 raw logits，Person logit 排除；权重采用 model criterion 原 gains，不另行选参。先对象均值，再类别均值。此 GT-positive reduction 区别于整图 all-anchor loss，也不是 Frozen Teacher KD，必须在真实 train 数据上审计尺度与覆盖。

`d_k = ell_k_poison - stopgrad(ell_k_clean)`；S2 `L_CNLP = mean_valid_classes ReLU(d_k)`。不强制模型参数或所有输出相等；保留原 Parent 的全部损失。Poison virtual update 用 create_graph=True，clean 每步 detach；没有一阶近似，也没有绕过虚拟训练的直接输入损失。这里“精确”指当前 PyTorch/native criterion autograd 图，native detached TAL 和 CIoU 内部约定仍保留。

合法无 NT GT 返回与 poison 图连通的零值。有效 NT GT 没有 positive 或 foreground evidence 属于覆盖失败，不允许无声忽略；当前没有引入 S1 budget estimator 的 independent-rematch 作为 CNLP 默认匹配。若真实覆盖审计失败，须单独记录 CNLP_DESIGN_CHANGE，再验证，不能改原 Parent。

lambda_CNLP、query/support batch、最终 reduction/归一化和 J 均未锁定。lambda=0 只允许 Parent 等价探针，正式锁要求正且有训练标定 SHA。该模型使用原始 reference 初始化；选中 Parent victim last.pt 只是评价来源，不用于初始化。
