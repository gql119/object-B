# S2-CNLP 正式训练报告

状态：COMPLETE / VERIFIED / ARCHIVED；2026-10-16。S3尚未启动，本报告不代表S3结果。

RunID：`s2_cnlp_s0_eps8_person_seed0_20261015_v1`；source `017591686efb3675ee8d55debf1e2cd7da4e0e17`；Parent固定S0。

已完成原始24epoch/9144Adam更新，实际69,902.03118577832秒（19小时25分02秒）；16551train、4952cleanval、6095目标图、13256PersonGT像素/标签审计通过。随机初始化YOLOv8n seed0训练完整200epoch，训练记录耗时10847.1秒（约3小时00分47秒）。last.pt为主评价，best.pt为次评价。没有重复训练或评价。

| 方法 | Person AP50% | NT mAP50% | NT mAP75% | P | U | S | WCD | TCD3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean_last | 86.794322 | 76.646191 | 61.932586 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| Safe_Soft05_last | 2.899342 | 69.169809 | 51.589813 | 0.966595 | 0.902456 | 0.933425 | 0.281032 | 0.257851 |
| S2_last | 4.036846 | 68.912262 | 50.945288 | 0.953490 | 0.899096 | 0.925494 | 0.308426 | 0.274399 |
| S2_best_secondary | 4.763960 | 69.296483 | 51.248053 | 0.945112 | 0.904109 | 0.924156 | 0.290377 | 0.265216 |

主评价相对S0：Person AP50 +1.137504pp，NT mAP50 -0.257547pp，NT mAP75 -0.644525pp。当前单seed结果不支持CNLP提升目标抑制或总体非目标保留；不得按此结果调整S3锁定参数。

Person Recall=0、Precision=1来自validator全类别best-F1 operating point；不是固定置信阈值，不代表完美Precision或零AP。AP采用完整置信曲线。完整20类AP50/AP75/AP50-95/Recall/Precision保存在raw/comparison.json；CSV保留逐类AP50/AP75/Recall/Precision。

P=1−targetAP/cleanTargetAP；U=NTAP/cleanNTAP；S=2PU/(P+U)，无静默裁剪。WCD是19个非目标相对AP50损失最大值，TCD3是最大的3类均值；S2最差类dog。单seed工程观察，不作显著性/因果/背景吸收/Person-to-Dog错误消除结论。

噪声总损失epoch1=1.1256108214，epoch23=1.0471456403，epoch24=1.0464058707；epoch24平均CNLP=0.0017745453、加权=0.0001037852。日程完成不等于收敛，独立victim结果不证明机制因果。

最终last.pt SHA `0ddaed6893b4824d4f7fd4c27854719b46f1d172e9b06d207908cdf6fc3e0d24`；best.pt SHA `9579b95a38ac8ace8d3f4d54da6bbc19f42a3c3d296a52b74eb5e433d89500e8`。两个checkpoint的Ultralytics epoch字段均为-1（strip optimizer语义），不伪称其内部epoch=200；连续results1..200、嵌入200行历史、正式source以及评价SHA共同证明last@200来源。

最终噪声checkpoint SHA `83cf11efac04a85a4e476c497cd55a2af56d4bb4b15a38838374ad9984f4af40`；Adam9144/scheduler24、完整RNG/CNLP/query状态已CPU只读核验，206保存点保留；Frozen/Adaptive不变。配置锁SHA `d408ee2d3a2cbd3b1945c64d414fd824f4d40e53db61899c7f3b4d65967b3b01`。

归档：docs/s2_archive/raw/（rrctl校验9项正式报告，另校验下载4个原始evaluation JSON和results.csv）、LOCAL_ARCHIVE_MANIFEST.json、LOCAL_ACCEPTANCE.json、s2_final_verification_20261016.json。未传输数据或权重。

控制面：直接pull曾因尚未执行rrctl wait完成确认而缺artifact_manifest；没有实验失败或GPU重跑。随后既有rrctl wait完成artifact contract并pull成功，terminal completed/exit0。periodic检查对已正常退出进程的暂时告警由completion healthy/complete消解。

Next：GPU空闲且目录无冲突时，按已审查的原始S3配置与原Parent初始化启动一次S3；不从S2扰动初始化。保留原beta/tau/lambda/预算/24epoch/200epoch设置。S3完成后统一比较；GitHub发布仍待对应完整协议及新分支政策，当前未发布本轮新结果。
