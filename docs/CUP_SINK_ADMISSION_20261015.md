# CUP 实际计算作用准入（2026-10-15，GPU观察前冻结）

保持J/innerLR/lambda、CUP beta/tau/clip/bounds提案不变。正式S2/S3仍使用完整train manifest和同一query sampler；下面只为有界计算作用探针。

原固定38case已知存在多类别query。按sha256('CUP sink seed0 '+stem)选择前4个、排除实际固定32张support：2010_005930、2011_000098、2008_008177、000632。只按训练身份和已有GT多类别覆盖选择，不看validation AP或梯度好坏。

执行新的同一生产优化循环10步/EOT2；CUP历史从全零开始，不能预填人工损伤或AP权重。每次success outer optimizer后提交一次EOT平均历史。比较实际CUP聚合与同一query同一反事实damages的uniform聚合在原Fourier变量上的梯度；两者均从真实虚拟训练图求导。检查所有20EOT的weights sum1、相对uniform[.5,2]，并要求至少一批加权/均匀梯度差>1e-12、CUP steps=10。若无作用则门槛不通过，不据此修改tau/样本/权重。

与Parent等价/S2/S3连续续跑联合准入共40optimizer steps/80EOT，另12个matching edge batch，总92production batches<100。虚拟SGD次数另外报告，不能混为outer updates。此探针不证明效用改善、稀有类AP保护或Person-to-Dog FP被消除。

额外记录full target attack gradient（BG_sel+rel_sel+lambda_adapt*L_adapt_target）与CNLP梯度余弦及加权CNLP/Parent范数；补足早期unit-gradient只用BG+rel代理的范围限制。只测量，不改变任何目标或权重。cup_sink的固定4query子集单独绑定到checkpoint identity/receipt，正式完整query sampler不改。
