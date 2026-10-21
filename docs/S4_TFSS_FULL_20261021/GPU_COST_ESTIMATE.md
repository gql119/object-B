# GPU时间规划估计（尚非Full实测）

精确预算保持：noise24epoch、6095目标图、batch16、EOT2、9144实际Adam更新；victim200epoch、batch36、workers16、AMP=True；另有一次归因缓存构建、完整数据导出、last/best clean-val评价。

总时间应按
`T = T_clean_maps + 9144 * t_noise_step + T_export + 200 * t_victim_epoch + T_eval(last,best)`
计算。当前尚未获准在正式教师上进行GPU smoke，没有可靠的Full步耗时，不能把单次CPU组件耗时当作GPU估计。

供排期使用的**假设场景**：若热缓存noise每更新2/5/10秒，则9144更新分别约5.08/12.70/25.40小时。它们不是测量值，不包含归因/导出/共享服务器竞争。

已有鲁棒性MeanF成功观察在约184epoch时累计13691.6秒，约74秒/epoch；仅作相同victim协议的排期参考，可暂按4–6小时/200轮预留，不承诺新数据/资源竞争下等速。于是noise热缓存与victim相加的条件规划范围约9–32 GPU小时，冷归因、导出与评价另加，实际区间可能更宽。

等旧流水线完整验收、队列清空后，先用本候选2个实际GPU Adam步骤记录cold-map耗时和step耗时，再更新估算。不会为缩短时间改变batch/workers/epoch/eta/优化器/训练超参，也不会据代理loss选择新参数。
