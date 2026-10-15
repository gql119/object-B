# Matched Person proxy 检查（GPU观测前记录）

用户任务要求检查明显Person suppression proxy退化。原lambda0等价探针使用手动固定16交替batch，S2/S3生产循环使用同一32图np permutation，二者不能直接跨不同batch比较proxy。新增单独15步/30EOT有界探针，不复跑已完成的GT覆盖或恢复检查。

Parent lambda0、S2、S3各从原参考checkpoint/原Fourier初始化/seed0独立运行同一生产循环5步，使用同一固定32train图和同一完整train-query sampler。逐EOT记录实际support/query IDs，要求三个路径完全一致。仅比较target_score_clean_after均值、中位数及相对变化，并报告真实lambda。无AP或参数搜索，不据此自动改lambda/CUP。checkpoint/PNG/数值要求保持，正式全noise/victim关闭。该短程proxy不能替代最终Person AP/Recall。
