# S1-CSB 正式结果（2026-10-15）

状态：COMPLETE。24/24噪声epoch、9144/9144step、完整16551train/4952cleanval审计、随机YOLOv8n200epoch、last主评价与best次评价全部完成；rrctl退出0并完成artifact contract。

| 方法 | Person AP50 (%) | Person Recall (%) | NT AP50 (%) | NT AP75 (%) | All AP50 (%) | P | U | S | WCD | TCD3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean_last | 86.794322 | 75.817138 | 76.646191 | 61.932586 | 77.153597 | 0.000000000 | 1.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| Safe_Soft05_last | 2.899342 | 0.066254 | 69.169809 | 51.589813 | 65.856286 | 0.966595258 | 0.902455927 | 0.933425073 | 0.281032380 | 0.257851168 |
| S1_last | 9.167047 | 0.397527 | 70.457610 | 53.590229 | 67.393082 | 0.894381949 | 0.919257822 | 0.906649287 | 0.270314875 | 0.218793131 |
| S1_best_secondary | 10.213072 | 0.375442 | 70.734623 | 53.595717 | 67.708545 | 0.882330180 | 0.922871992 | 0.902145836 | 0.266415461 | 0.214350264 |

S1相对S0：Person AP50 +6.267705 pp，NT AP50 +1.287801 pp，NT AP75 +2.000417 pp；S变化-0.026775786（无量纲）。Person超出1pp容差且高于5%研究参考线，因此按预注册规则选择S0。NT提高/WCD降低不能覆盖该规则。没有使用best替代last。

噪声实际耗时35752.084494秒（9小时55分52秒）；victim results.csv累计训练时间10386.30秒。完整日程不等于已有收敛判据。Frozen/Adaptive初始化均保持不变。

预算机制汇总（完整6095图；图均值/像素加权分开）：
- epsilon_mean: 图均值6.695398 byte levels；有效图数6095。
- high_sensitivity_budget: 图均值2.250374 byte levels；有效图数4230。
- low_sensitivity_budget: 图均值7.815914 byte levels；有效图数6083。
- overlap_budget: 图均值4.999722 byte levels；有效图数3835。
- nonoverlap_budget: 图均值7.059215 byte levels；有效图数5722。
- delta_budget_mean: 图均值-0.098468 byte levels；有效图数6095。
- 像素加权Person预算7.026671 byte levels；像素加权重叠预算5.539361。
- 无NT图1715，含NT图4380；8337 NT GT中native遗漏12，恢复12，最终遗漏0。

高敏感区域预算低于低敏感区域是构造行为证据，不是因果有效性证明。名义/平均预算与实际扰动幅度不同；S0原重叠幅度缩放与S1空间裁剪会改变有效强度。只有短程uniform诊断，无完整等强度control victim，不能把NT收益归因于空间分配本身。类别结果见VOC20 CSV；当前未计算按GT大小的预算/扰动分层，应记未测，不能猜测。

AP依据完整置信度曲线；Recall/Precision是validator全局best-F1工作点，不等于固定阈值召回或AP推算。单seed0对比不能给统计显著性结论。

S0 archive GitSHA0d43da4并非同时期runtime commit，历史runtime GitSHA缺失；采用45源文件/config/artifact SHA核验。S1 runtime source16e0733，最终lastSHA 3f773f5cc6986ce567459ac3dbc75c9b5419dc12861c59e1de85f24da8813fda。

原始小型JSON/CSV位于docs/s1_archive/raw；verification_20261015.json及s0_history_verification_20261015.json记录协议/状态/预算汇总。20类AP50/AP75/AP50-95/Recall/Precision全部保留，未重新训练任何实验。
