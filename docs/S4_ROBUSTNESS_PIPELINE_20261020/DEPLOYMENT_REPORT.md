# S4服务器独立流水线部署报告

核验时间：2026-10-20T19:37:48.602595+08:00（香港时间）。

唯一 Pipeline RunID：`s4_robustness_server_pipeline_seed0_20261020_v1`。控制器 PID=2405363，start ticks=311657949。
RRCTL 已实际启动；tmux session=`s4_robustness_server_pipeline_seed0_20261020_v1`，原启动连接退出后通过新的连接复查仍存活。真实重复启动测试返回 existing_instance=true，前后控制器身份不变。
调度器源码 `d84c28c731ae10c6a0f32d5ed86ac5635e1a0a8c`；实际科学源码保持 `e17c24c54aae4cdae4bcc4703facf2a57052d2c8`；S4鲁棒性锁 `00af2dc40c5c4d263246e78f70195bf8a53a16b9118bb289a9b307d93bafdb91`；原S4配置33d3d7…未改。

## 当前阶段及衔接

服务器已经自主进入 Gray；active_job=`gray:S4:attempt01`。JPEG50/JPEG30均已通过服务器完整验收，没有重复训练。
实际Gray S4子进程 PID=2422613、start ticks=312055320，当前UID与启动身份一致；同根目录正式训练子进程数=1。
Gray S4子任务已启动并核验身份；尚未记录训练epoch，处于训练前准备阶段。
Gray全量数据变换/审计尚未在这次启动快照中完成，不提前宣称数据审计通过。

之后的衔接全部由服务器执行：Gray → BDR → Phase A验收与报告 → Gauss → MeanF → MedianF → Data Aug → Phase B验收/最终报告 → 归档退出。Codex或两小时心跳都不会发送下一阶段启动命令。

## 参数、对照和数据身份

BDR已锁定5bits、31levels、float32单位区间、NumPy round ties-to-even。其来源与既有真实batch Smoke和G1数据/加载器等价性证据均已在服务器重新核对。
四个Phase B历史Clean200对照（Gaussian3/sigma0、Mean3、Median3、native degrees10/shear2）实际checkpoint/hash/参数/200行CSV核验全部通过；重评原AP50/Recall容差1e-7。若来源或参数无法等价，另建匹配Clean200；若重评指标不一致则停止并保留证据。
Poison仅使用原S4-TFSS-Utility完整9144update数据。所有变换只作用训练集，标签/尺寸及4952clean验证保持既有协议；所有新Victim使用seed0/randomYOLOv8n/200/batch36/workers16/AMP/nativeSGD，last@200主评价、best次评价。AT=NOT_RUN。

## 后台、恢复及限制

无需Codex、VPN、SSH控制终端保持连接；RRCTL detached tmux供应后台生命周期。CPU合成调度Smoke真实经RRCTL运行并exit0，验证了队列衔接、flock单实例、已完成阶段跳过、控制进程退出后同一独立supervisor/child继续并被重新接管；它没有GPU前向、训练batch或科研指标。
服务器重启后不能保证自动拉起：当前user systemd degraded、Linger=no，未修改系统/用户服务。状态持久化，允许核验原进程确实结束后用同一配置恢复；不声称已实现主机重启自启动。
环境失败最多重试2次，退避60/300秒；未知或科学正确性错误停止。完整200行/原生嵌入200行与快照可证明时只补评价；否则保留失败产物、新恢复RunID、fresh seed0完整200，明确不声称DataLoader/prefetch精确续训。已完成训练臂及已接受阶段不重跑。
资源沿用既有阈值，CPU/GPU忙和load为速度记录；RAM32GiB、GPU空闲显存24000MiB、disk150GiB和三窗口持续I/O>=20%守卫保留。活跃自身GPU不套用空闲准入。只对身份与稳定checkpoint核验通过的本任务进程执行安全暂停/恢复，从不干预他人。

## 状态与日志

输出根：`/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1`

- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/pipeline_state.json`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/pipeline_progress.json`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/pipeline_events.jsonl`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/pipeline.log`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/resource_latest.json`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/resource_admission.json`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/jobs/<job>/stdout.log`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/jobs/<job>/stderr.log`
- `/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/outputs/s4_robustness_server_pipeline_seed0_20261020_v1/reports/`

服务器只读状态命令：

```sh
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -B /project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/repos/s4_robustness_server_pipeline_seed0_20261020_v1/scripts/run_s4_robustness_pipeline.py --config /project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_20261019/repos/s4_robustness_server_pipeline_seed0_20261020_v1/docs/S4_ROBUSTNESS_PIPELINE_20261020/PIPELINE_CONFIG.json --status
```

本地保存同一条命令：execution/PIPELINE_STATUS_COMMAND.txt。该命令只读取状态，不创建训练或接管任务。

## 预算与交付

仍需Gray/BDR各S4+Clean200，加四个Phase B S4Victim200，共8次正式训练；另对4个历史Clean进行last/best验证集重评。已完成JPEG50/JPEG30不会重训。
已观测JPEG50/JPEG30每个200epoch训练约14,400–15,300秒（约4.0–4.3小时）；8次训练约32–34小时，连同数据处理、审计和评价预估36–48小时，资源竞争或等待可能进一步延长。这是估计，不是完成时限。
预计新增数据/保留checkpoint约100–160GiB；部署时磁盘空闲约1.12TiB，容量守卫继续生效。每臂最多2次环境恢复会额外增加成本，全部失败证据保留。
阶段报告、完整VOC20、P/U/S/WCD/TCD3、CSV200/checkpoint SHA、源/配置/数据归属均由服务器验收并归档；最后生成reports/PIPELINE_ACCEPTANCE.json=COMPLETED_AND_ACCEPTED后正常退出。AT不执行。
唯一两小时Codex监控已改为只读状态通知，FREQ=HOURLY;INTERVAL=2；它不再参与阶段调度。最终小结果可通过原RRCTL绑定取回。大型数据和权重始终留在服务器。

## 部署验证证据

- execution/PIPELINE_DEPLOYMENT.json：实际绑定/PID、独立连接复查、正式唯一实例重复启动测试。
- execution/PIPELINE_BOUND_SCIENCE_PROBE.json：科学源码/固定配置/既有GPU Smoke/四历史Clean/BDR服务器核验。
- execution/s4_robustness_pipeline_supervision_smoke_seed0_20261020_v2_CHECK_LATEST.json：completion health→wait exit0→小结果取回。
- execution/GRAY_SERVER_AUTOSTART_VERIFIED.json：存在时才证明Gray由服务器自动启动。
- 调度Smoke v1在运行前因继承JPEG30会话名而被RRCTL碰撞保护拒绝，无新workload，旧JPEG30未受影响；原失败证据保留，v2使用独立会话。


补充核验：logind有效D-Bus属性KillUserProcesses=false；正常用户会话退出不清理此后台进程。只读状态CLI已在实际部署路径执行并解析通过，实际audit/report源码字节SHA与提交内容一致。
恢复边界：可接管仍存活且有退出/身份凭据的同一子任务。主机重启等情形若使控制器和子任务同时消失且没有退出回执，会安全BLOCKED，需人工核验并记录恢复；不是重新运行一条命令就承诺自动续训。自动有限重试只处理已识别的阶段环境故障，未知控制器或科学错误不自动忽略。
USER_TASK_PROVENANCE.json区分原附件/本地精确副本SHA与Git仅CRLF→LF规范化后的任务文本SHA；规范化后逐字节相等，运行配置保留原附件SHA，未改运行配置。SOURCE_REUSE.json使用实际提交/服务器源码字节SHA，并保留Windows副本哈希。
