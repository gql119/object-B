# COCO S4 方法一致性报告

**Original S4 — COCO80 Dimension-Compatible Implementation**

原VOC generation b1e29510e7b1ed5fa0fe361a27132d8790e65c14；Victim AMP ab626917f20917e58cf30033ac99df00938cb561。本轮修复恢复原计算入口，不重新设计方法。受保护8个源文件中7个与原SHA完全相同，Safe-BG文件匹配已授权dynamic-D差异；另有b2_semantic已授权维度适配。55项相关回归0失败/错误/跳过，VOC24同设备对照数值精确；COCO39独立P3=48。

- 严格TSFL保持原39索引/分数/排名/校准身份；同一min(ceil(.15*C),eligible)规则，.4/.6/top3/.15/min4未变。
- Safe-BG all79/rank8/alpha.5/active3/同序SVD/范数恢复/有界缓存保持；实际55个缓存均rank8，CUDA最大Gram 2.384185791015625e-07；原阈值1e-5不变。
- Original support1/.5/0→8/4/0，STE2.01/255、TFSS eta.85与Qt/Rnt公式、原始JND/Fourier64候选→16保存、EOT2、Frozen/Adaptive1步、Adam24/seed0、全部损失权重不变。
- 真正使用canonical S4Trainer/SoftSpatialProjection/CleanMapCache/optimize_s4。每个EOT先TFSS加权，再原始sharedEOT/letterbox供Frozen；Adaptive使用原始对应的加权poison。实际对Fourier叶参数求BG/rel梯度，未写入或改变optimizer.grad累积。
- 原始训练的随机幅度缩放/dropout与最终确定性导出不同，属于原方法；验证用同一最终生成上下文比较，未关闭训练增强。
- 实际保存16x3参数→原生generator→同一TFSS图→PNG重新解码，在CPU/CUDA逐元素一致；单图最大变化2字节、重叠1、框外0，均在8/4/0预算内。只是有界产物，不是25K正式毒化数据。
- Teacher200及单独5K评价、完整25K校准/K8/all79/13630支持清单均哈希复用；不重复训练或校准。当前无Victim AP，Teacher AP不作Clean200分母，LPIPS未测保留null理由。

额外导出回调设备证据状态：passed。完成前不宣称全线最终验收；全部实际复验通过后，依已记录人类条件授权可建立新RunID正式启动。
