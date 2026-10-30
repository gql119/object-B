# S4 AT 直接终止、CUDA比较与正式重启回执

状态：FALLBACK_ORIGINAL_PROTOCOL_STARTED。更新时间 UTC：2026-10-30T10:50:43.605450+00:00。

## 旧实验终止与保留

旧 RunID `s4_at_serial_seed0_20261029_v2` 已于2026-10-30T05:23:23Z通过本人专属 RRCTL 终止，真实终态为 ABORTED。终止前epoch87，86个完整epoch，39666 batches、20265次SGD、12次合法AMP跳过。epoch86完整checkpoint SHA256为 `925dcaf995a6880d987e218b1e28a72c6008a93bbcf0cb26e10001167ea8e3ee`，日志、失败、锁、终态全部保留。旧进程组及CUDA上下文已消失；另外三臂当时未启动。旧checkpoint不作为新训练输入。

精确终止证据：`execution/direct_stop_20261030/20261030T052322Z_OLD_AT_TERMINATED.json`。单次控制安全审查负面LIFECYCLE-01与主审修复闭环均保留；没有第二次控制审查。新队列CPU真实控制验证通过：严格串行、重复控制器拒绝、旧全局锁SHA不变。

## 实际CUDA优化证据

Candidate02/03各完成CleanAT4零容差组件：每侧16个真实native outer、10次SGD、6次合法AMP跳过；每一步PGD梯度/TAL/loss/扰动及外层模型、optimizer、AMP、BN、EMA/RNG哈希一致。它们未完成所有四臂与resume全套候选验收，且没有被采用。

已完成8个实现ABBA测速样本与4个Original Batch36/72 ABBA样本。下面是实际原始样本，不能当作通过资源可比性验收的加速排名。

| 实现 | Batch | 原始 images/s 范围 | 原始秒/batch范围 | 分配显存峰值 GiB |
|---|---:|---:|---:|---:|
| original | 36 | 21.055–22.562 | 1.596–1.710 | 7.939 |
| 02 | 36 | 20.966–22.347 | 1.611–1.717 | 7.899 |
| 03 | 36 | 22.880–22.925 | 1.570–1.573 | 7.899 |
| original | 72 | 22.897–23.099 | 3.117–3.145 | 15.813 |
| original | 48 | NOT RUN | NOT RUN | null |
| original | 64 | NOT RUN | NOT RUN | null |


Candidate02/03 ABBA均为INCONCLUSIVE：原锁定时钟、CPU负载、热/功耗限频占比条件未通过。Batch72 ABBA同为INCONCLUSIVE：时钟波动及ROI遥测不足。validated speedup百分比为null。首次bo1实际遭遇另一个GPU PID，已作废并保留；未操作该进程。没有放宽容差或将batch翻倍当作吞吐翻倍。

Batch48、64及CUDA事件分解未运行；限定派发窗口已过，停止扩展搜索以执行用户明确授权的回退。不能据此宣称它们不可行或更慢。没有可靠的全候选最优排名；最终采用通过原科学协议验收的Original Batch36，不采用Candidate或新Batch。PGD10、4/8预算、步长1/2、640、workers16、200epoch、nbs64、native SGD/AMP及完整GT保持原协议。新Batch的accumulation/BN/warmup/update/EMA差异仅作为未采用方案披露，实际原协议weight decay为0.0005625000000000001。

## 冻结当前源码预检与科学门槛

运行源码：`87456fa7a57206551b0e3be0253cc4866f0cdcf1`，已正常推送并读回远端SHA。19个科学文件与60个OriginalS4来源字节一致51ce已验收版本。GT/数据复用已接受清单，未重跑COCO、S4生成或全量导出审计。

当前源码真实四臂worker CUDA预检通过26个native outer，各臂均发生SGD，跳过数依次6/6/5/5，loss及状态有限；四臂新随机初始化SHA `0ff81f5755379765bfb1c7080b73178b88da200c1cdd2fd9f551cadd2e3b9eb6` 一致。独立DetectionModel基线与实际worker loss均453.893798828125，diff0/tolerance0。当前源码61个outer的完整epoch安全相位恢复通过，model/optimizer/scaler/scheduler/EMA/gradients/PGD RNG及attempt相位零差；不声称workers16预取轨迹逐位恢复。新正式训练仍从epoch0开始。

预检RunID：`s4_at_selected_native_20261030_gpu01`、`s4_at_selected_baseline_20261030_gpu01`、`s4_at_selected_phase_20261030_gpu01`。均完成exit0、completion health、wait与小文件pull一次；无并发本人GPU。Git读取发生两次TLS EOF，保留连接失败，并通过已连接GitHub官方API准确读回同一分支SHA后继续；未修改TLS/VPN/代理。

唯一新科学审查：scientifically_correct / allow_run。原始JSON与Markdown保留，正式启动还须真实科学闭环与不可变门槛上传。没有把CPU验证或源代码相同当作GPU结果。

## 新正式四臂状态

新控制器RunID：`s4_at_serial_seed0_20261030_fresh_v1`。控制器是否已实际启动：True；是否已记录正式训练batch：True。顺序固定 CleanAT4 → PoisonAT4 → CleanAT8 → PoisonAT8；均为新随机YAML seed0、200epoch、Original Batch36。其余三臂等待同一控制器串行调度，不同时启动。

当前进度：`{"run_id": "S4AT4_Clean_seed0", "epoch": 1, "completed_epochs": 0, "at_batches": 44, "actual_updates": 38, "legal_amp_skips": 6, "loss": 455.87371826171875, "UTC": "2026-10-30T10:49:19.364254+00:00", "actual_shape": [36, 3, 640, 640], "actual_PGD_steps": 10, "raw_devices": [["cuda:0", 0]]}`。
当前身份：`{"executor": {"pid": 3656364, "start_ticks": 398177890, "uid": 1014, "alive": true, "expected_start_matches": true, "ppid": 617924}, "queue": {"pid": 3656393, "start_ticks": 398177903, "uid": 1014, "alive": true, "expected_start_matches": true, "ppid": 3656364}, "scientific_GPU": [{"pid": 3656582, "start_ticks": 398178502, "uid": 1014, "alive": true, "expected_start_matches": true, "ppid": 3656393, "cwd_matches_release": true, "parent_matches_queue": true, "run_id": "S4AT4_Clean_seed0"}]}`。
日志：`/project/guoqinglong/projects/s4_at_robustness_20261027/outputs/s4_at_serial_seed0_20261030_fresh_v1/result/S4AT4_Clean_seed0.console.log`。

新RunSpec/READY/launch及门槛见 `execution/direct_stop_20261030/s4_at_serial_seed0_20261030_fresh_v1/`、`SELECTED_RUNTIME.json`、`SELECTED_GATE_UPLOAD_VERIFIED.json`；仅在实际文件存在时代表门槛已闭合。旧全局锁/终态未清理，采用经CPU验证的新独占flock与全新输出根。当前GPU若新CleanAT4已启动，则由其使用。

## 后续与结果边界

保持同一个ACTIVE监控器，禁止重复控制器、并发测试、热替换或旧checkpoint续训。四臂真实200epoch后核验连续epoch、last.pt@200、完整状态/optimizer/scaler/EMA、初始化/source及验收回执，复用各臂已计算的完整cleanVOC4952/all20 last主/best次评价，完成匹配预算全类别P/U/S/WCD/TCD3与最终发布。目前没有新的正式AT效果指标，不作性能或因果效应结论。COCO已接受并发布，保持原样。
