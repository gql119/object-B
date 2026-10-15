# CNLP train-only calibration：当前已验证状态

Parent=S0，完整S1归档与原预注册1pp规则见parent_selection_report.md。所有新标定正式生成关闭、victim关闭、outer optimizer更新0，使用固定train-only support/query计划；并非最终检测效果或收敛证据。

| RunID后缀 | J | case/EOT | 耗时s | median clean-query learning gain | 非零mean-EOT CNLP梯度case | 与BG+rel attack proxy强冲突 |
|---|---:|---:|---:|---:|---:|---:|
| J1_train38_seed0_20261015_v2 | 1 | 38/76 | 215.591918 | 0.0000772476196 | 21/38 | 0/21 |
| J3_train38_seed0_20261015_v2 | 3 | 38/76 | 323.912117 | 0.0000849962234 | 23/38 | 0/23 |
| J5_train38_seed0_20261015_v2 | 5 | NOT_VERIFIED | null | null | null | null |

38个case对应33张unique query图像，累计67个非目标GT case-observations；有重复图像，不是38个独立样本。所有RunID前缀为cnlp_unitgrad_。源码ca57a71d2a934946fe983620aefa2e41da8efc0a；fixed plan SHA b513e553afa03af6eff550eeefe8d5bdc94ddb6e9f7c6fbde70c4375657fa780；train manifest SHA7ea21929d1e1d89aff8e345ed1c7ddd58367c0bad4c9f71574487a19e33a831f。J1/J3 Frozen/Adaptive/Fourier均未改变。

J1 v1源码6f757c9已完整计算，但只记录各EOT梯度范数，不能用作预注册mean-vector梯度标定，故全部保留，CODE-004仅补齐诊断记录后使用fresh v2输出；不更改损失/样本/科学参数。新增3项CPU测试验证向量抵消、非平行均值方向、非finite拒绝。

J5 v2已唯一启动（tmux与RunID同名，PID391318），观察器随后SSH port22超时。尚未读取终态，不代表计算失败，也不能标为通过。不得重启或重复标定；恢复连接后inspect/wait/pull已有RunID。

即使J1/J3初始门槛通过，仍须取得J5、按预注册规则选J/lambda，进行12个历史训练GT边界覆盖、真实canonical Parent等价、S2/S3生产5步与3→5续跑/像素PNG、CUP实效与类别历史审计，再调用一次最终科学审查并锁定两协议。正式S2/S3、独立review pass、GitHub发布均NOT_RUN。

后续两个新入口代码仅syntax检查，尚未提交/部署；counterfactual global export元数据更名后，相关synthetic完整/断点/完成边界恢复用例1passed。原Parent45源码文件再次SHA一致。所有失败和临时产物保留，没有删除文件。


## 连接恢复后核验：2026-10-15

J5 v2已正常完成38case/76EOT，duration442.690727s，peak8,369,923,072bytes，median clean learning gain0.0004137158394，20/38非零平均CNLP梯度，强BG+rel proxy冲突0/20。Frozen/Adaptive/Fourier未改变，outer update仍0。此前SSH超时没有造成计算失败，未重跑J5。

三组均过预注册门槛，选择最小J=1。候选lambda=0.058485507823134296由全case25%梯度上限项决定（典型10%参考项为0.81175321079131，不能据此放宽规则）；非零case的加权norm/Parent norm中位数0.0066853987351454085，max0.25。不是所有后续训练query的全局梯度上限。CUP候选beta0.95,temp0.0003758072853088379,clip0.0010146975517272938,相对uniform边界[0.5,2]。仍须真实匹配/生产恢复/CUP sink/独立科学审查；不是正式锁。

全部原始小型标定JSON与派生分析已归档docs/cnlp_calibration/。configs/s2_s3_train_proposal_20261015.json保持formal/victim disabled。原38case=33unique query/67NTGT case-observations，不是38个独立样本，不作统计显著性或最终AP推断。
