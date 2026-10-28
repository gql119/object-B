# COCO Clean200 正式恢复回执（2026-10-28）

## 当前实际阶段

原唯一串行控制器已在 **2026-10-28T12:44:18.199663Z（香港20:44:18）** 启动正式 `clean200`，最新12:45:12Z观测为RUNNING。控制器PID2334114/start369629083/UID1014未变；新正式子进程PID3072365/start381613294/UID1014存活。未创建第二个控制器、监控或竞争性GPU任务。

后续12:47:01Z（香港20:47）只读原生CSV尾行确认：epoch1，214个完整训练批次、208次真实SGD、6次合法AMP跳步；损失451.4182，模型/EMA/优化器/损失均有限，scale1024。正式初始tensor SHA与GPU04一致；尚未完成第1个epoch，不声明Clean200完成。

- Workflow：`s4_coco_experiment_a_seed0_20261027_risk_v1`
- 原RRCTL索引：`s4_coco_experiment_a_seed0_20261027_risk_v1_transport02`
- 当前实际源码：`89f4f72b380eabd46dc1992adfc195e58e15e9fa`
- 隔离Release：`releases/coco_victim_clean200_20261028_gpu04`
- Release SHA256：`1806139de4954889022c3ca96bd183d6a6830b91b8b78a4810065fb95de51193`
- phase配置SHA256：`7e53ecf579a066319c71474980a4fb3e748ca7e764148efe3cf61021b84f7f89`
- READY SHA256：`b37660d4b716f3375dde47450726900d7842ad4d0b89cab59d09dee7b3117275`
- 正式日志：`/project/guoqinglong/projects/s4_coco_experiment_a_20261025/outputs/s4_coco_experiment_a_seed0_20261027_risk_v1/jobs/clean200/console.log`
- 正式输出：同workflow目录下`clean200/`；不是隔离GPU探针输出。

## 已验收前序与复用

Fourier24原控制器及独立审计已确认24epoch、20448次Adam、每epoch852次；参数/优化器/调度器/RNG/正确S4Trainer路径/40896次TFSS-EOT-Frozen-Adaptive接线已验收，没有重优化或续用历史错误12步checkpoint。

Poison25K正式导出于11:03:05Z验收完成。独立最终审计核对25000新图像、25000标签及全部输出SHA、181475GT、80类、13630Person图、11370noPerson图；标签字节及noPerson原JPG字节身份一致。逐文件生产decoded8/4/0预算记录绑定独立核验的未变图像SHA。导出receipt SHA `7c156811ba3f422eee3dc5cab35d598b858d23cccabeb4d925bba9766360817f`。没有重导出、重新校准或重训Teacher。

## Clean预检失败与最小修复

三个失败RunID及日志、源码、退出状态完整保留，均未复用：

1. GPU01：旧Teacher提交硬编码的工作流绑定与实际5c49控制器不符。96eb5f8改为使用既有不可变workflow_admission；保留PID/start/slot约束，真实CPU正例及7种坏绑定拒绝通过。
2. GPU02：从teacher错误导入attach_safety_callbacks。4a3e9bd改为真实owner `ue_framework.s4_victim`；21个生产导入及原配置契约CPU通过。
3. GPU03：原生CUDA两批损失有限，但AMP合法回退导致0次SGD，正确被至少1次实际更新门槛拒绝。保留该失败；未降低AMP初始scale、关闭检查或修改损失。

89f4f72将有限探针预算显式参数化为1–100，本次16批；正式200epoch超参数未变。同时修复正式初始化回调额外读取训练批的问题：正式路径仅保存初始模型身份，真实CPU验证0批/0forward、模型BN和RNG不变；独立参考模型/损失对比仅在有限探针中运行。

## 当前实际CPU/CUDA与独立审查

GPU04在89f4f72生产入口、batch36/img640/workers16/AMP/原生SGD/seed0上执行16个真实批次，10次实际SGD、6次合法AMP跳步；损失/模型/EMA/优化器全部有限，0验证前向。退出0，四个小报告拉取一次；所有5个checkpoint保留在服务器，没有上传权重。

CPU只读主审计核对154个发布源码与7个原生文件SHA、原生AMP16行、checkpoint epoch0/optimizer/scaler/EMA、初始tensor身份。另5个原生数据/损失文件服务器哈希与本地已有源码精确一致。自动审批拒绝额外下载服务器源码后，改用已批准的哈希比对，未传输该源码内容。

独立原生损失对比两侧均487.11358642578125，差0，既定容差0。证据边界：baseline原始ID未单独保留；AMP日志row0是随后训练批，不能当作baseline样本ID。正式路径不运行该额外baseline取样和参考前向。

按pre-run-implementation-review技能，恰好一次独立Clean科学审查 `/root/coco_clean_victim_scientific_review_20261028` 返回scientifically_correct/allow_run，无阻断项；原JSON/MD保持不变。原JSON SHA `0ef9a0b7c25b19fddc44190560ad4e62590817b34ebef8f66fbfbbccae207b22`。主闭环及当前发布/依赖/资源门槛通过，所有Release、review、closure及READY上传后逐一SHA读回，原控制器消费READY并启动。

## 发布与后续

当前源码已正常非强制推送。ls-remote读请求存在TLS EOF；正式登记采用同一分支一次新鲜成功的正常push协商，明确返回精确HEAD→远端ref `[up to date]`，同时核对干净已提交HEAD与所有manifest字节，作为等价的实际发布证据。没有降低TLS、修改VPN/代理或豁免未发布正式源码。用户“提交失败直接启动clean探针”的例外仅用于此前有限探针；正式发布门槛仍真实满足。

正式Clean为独立随机YOLOv8n80类YAML、pretrained=false/resume=false、seed0、200epoch，与随后S4 victim保持相同协议和初始tensor身份。主评价last.pt@200，best仅次要；训练期间不用val，不把历史Teacher当Clean分母。S4 victim及完整clean val2017 5K/all80评价尚未开始，待各自真实阶段门槛通过后由同一控制器继续。AT按用户最新指令推迟。

科学标记保持 **Original S4 — COCO80 Dimension-Compatible Implementation + Level C NT-risk Boundary Extension**，协议`coco80-ntgt-native-tal-then-gt-risk-v1`，不宣称严格Original S4全局等价，不使用S4-Full。尚无正式victim AP/效用/因果或显著性结果。

小报告与原失败证据归档在同目录`CLEAN200_RECOVERY_20261028/`。权威实时入口为execution/COCO_CURRENT_STATE.json及COCO_EXPORT_CONTINUE_20261028.md最新追加；只读进度脚本observe_coco_victim_progress_20261028.py读取当前原生AMP完整CSV尾行，不修改训练进程。
