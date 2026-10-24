# COCO Experiment A 状态

更新：2026-10-24T17:21:38.197800+00:00。当前阶段：DATA_READY_PENDING_PHASE_VALIDATION。

本地Kaggle训练ZIP已成功转移学校服务器，双端SHA完全一致。完整train25K与原始clean val2017 5K图像及对应YOLO80标签均全量审计通过；正式数据根为`/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023`。数据下载阻塞已解除，无需用户再提供验证集路径。

Original S4 Safe-Soft05 + TFSS-Utility父提交`b1e29510e7b1ed5fa0fe361a27132d8790e65c14`，独立Victim组件`ab626917f20917e58cf30033ac99df00938cb561`。预算仍为8/4/0，全部原公式、损失、StrictP5/SafeBG/TFSS/FrozenAdaptive/Fourier及冻结优化协议不变，不使用Full。

已有45项CPU契约通过。生产入口检查详见COCO_PIPELINE_INTEGRATION_AUDIT.md：旧VOC入口不可直接用于COCO；Teacher严格train-only需要显式验证，单设val=False不能阻止原生最后一轮验证调用。后续还需完整源码/配置/数据绑定、有界实际GPU smoke、每个新科学候选的一次独立实现审查、fresh资源/无冲突准入、独立固定版本官方COCOeval与LPIPS依赖验证。

队列已恢复到数据之后的实现验证阶段；尚无正式COCO RunID/绑定，GPU启动0，未部署第二个训练调度器。正式流程仍为Teacher200→COCO统计/原型/79NT子空间/TFSS→原始S4损失链GPU验证→fresh Fourier24→全25K Poisoned及8/4/0审计→fresh Clean200/S4200→clean val2017官方评估；不能把数据就绪称为已开始训练。

Cutout/CutMix/MixUp与Full均已接受/归档/发布，不重跑。OP禁用并独立待协议批准。沿用唯一监控s4-robustness-existing-run-status；没有新增监控、环境安装或清理。

报告最终核验时间（香港）：2026-10-25T01:21:54.319857+08:00。

最新实现进度：Teacher train-only组件已完成两项定向CPU探针（两原生batch、1实际SGD更新、零val/EMA前向、完整checkpoint、5数据/15参数拒绝）。完整生产入口和服务器串行编排仍待完成；本地CPU环境与服务器CUDA环境不同，GPU/科学审查/正式绑定均未通过或启动。新自动启动授权见USER_AUTOSTART_TASK_20261025.md；详见TEACHER_TRAIN_ONLY_CPU_REPORT.md和COCO_EXPERIMENT_A_STATUS.json。
