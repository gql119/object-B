# 后继队列、资源和状态

当前准备分支 codex/s4-robustness-extended-defenses-20261022；runtime源码SHA以实际RunSpec和run_ref为准，不用后来文档HEAD替代。新root只从旧锁定父目录派生，不触碰旧root：`/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/b2_to_safebg_20261002/ue_project/ablation_safe_bg_20261004/candidate/robustness_s4_extended_20261022`。RunID `s4_robustness_extended_seed0_20261022_v1`；正式数据源仍为original S4，不是Full。

顺序：WAITING_PREVIOUS_PIPELINE → CHECK_PREVIOUS_ACCEPTANCE（自己的完整COMPLETED_AND_ACCEPTED收据）→ VALIDATE_NEW_CODE/source/native/data → 三个独立few-step GPU smoke（各<=42计算单元：Clean/S4各<=20真实batch +2个更新前loss forward；无正式validation/ingest）→ 一次科学审查凭据 → Cutout Clean200→S4 200→accept → CutMix Clean200→S4 200→accept → MixUp Clean200→S4 200→accept →部分报告归档→BLOCKED_PROTOCOL_APPROVAL。OP不阻塞前三项归档且不会被自动执行。

当前不新增GPU工作。server300s只读小状态轮询；网络中断不等于训练失败。单实例flock和job intent/PID-startticks/exit凭据拒绝重复；CPU监督smoke直接使用正式继承的run_job/systemd scope，无GPU/真实OOM注入。所有失败保留，不自动exact-resume、不重启原RunID，不更改epoch/batch/AMP/LR来绕过失败。

资源：同一GPU串行；沿用原三窗口capacity/I-O规则，RAM>=32GiB/GPUfree>=24000MiB/disk>=150GiB，持续全部3窗口IOwait>=20%阻塞新stage。CPU busy/load、GPU利用率/他人compute存在仅为速度遥测，不新增硬门槛。健康active own job不套空闲GPU门槛。无sudo、环境升级、进程干扰或原图写入。

资源预算（配置推算，未benchmark）：6次200epoch、每epoch ceil(16551/36)=460batch，共552000训练batch，另各last/best在4952clean val评价（共12次checkpoint评价），训练epoch自带原生validation亦保留。每次新增配对独立训练，无历史Clean分母。图像/GT使用只读链接，无6份全量图片；显存/CPU/RAM随冻结batch36/workers16/cacheTrue，不能根据猜测调整。保留last/best、每epoch安全副本/optimizer/RNG，磁盘开销需在运行准入确认；不自动清理。小时数尚未测，不能承诺完成时间。

只读状态命令（在本新工作树目录、使用既有RRCTL环境变量/解释器）：

`python -B -m robustness_s4_extended.control status --profiles ../s4_robustness_evaluation_20261019/execution/profiles.json`

执行器只通过 RRCTL Controller/ProfileStore/transport；新CPU控制run和后继run各唯一绑定。存在binding时仅status，不再launch。当前执行凭据和实际部署状态见execution以及reports/DEPLOYMENT_STATUS.json，不能把PREPARED描述成已部署。

审查门槛来自项目 `F:/autodl-tmp - 副本/ue_project/.agents/skills/pre-run-implementation-review/SKILL.md` 的 `isolated remote GPU few-step smoke -> one scientific implementation review -> official training/evaluation`。服务器能独立等待/训练/评价/顺序推进；科学审查凭据须由独立审查服务或现有唯一Codex监控签发，不能由调度器自证。服务不可用时保持PENDING_SCIENTIFIC_REVIEW，不需要本地持续连接维持server进程，但不会谎称无外部门槛。

实际直接只读查询（已从原F盘项目目录验证，无需改变ExecutionPolicy）：调用既有Python解释器并传入本工作树 execution/query_extended_status.py 的绝对路径。该脚本内置本机已审计RRCTL导入路径，仅通过transport读取小型JSON，不写远端health，不调用GPU、不启动任务。准备时的ps1包装被本机执行策略阻止，未修改策略，改用Python已验证成功。最终快照：Full第10/24轮、3519/9144更新；后继WAITING_PREVIOUS_PIPELINE，active_job=None。
