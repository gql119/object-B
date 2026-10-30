"""Derive a concise final handoff from actual small receipts only."""
import json,statistics
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DOC=ROOT/'docs/S4_AT_ROBUSTNESS_20261027'
SOURCE='87456fa7a57206551b0e3be0253cc4866f0cdcf1';TAG='s4_at_serial_seed0_20261030_fresh_v1'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def optional(p):return load(p) if p.is_file() else None
now=datetime.now(timezone.utc).isoformat();s=load(DOC/'CURRENT_STATE.json')
review=optional(HERE/'SELECTED_PROTOCOL_SCIENTIFIC_REVIEW.json');observation=optional(HERE/'NEW_FORMAL_OBSERVATION.json')
main=load(HERE/'CURRENT_GPU_MAIN_AUDIT.json')
started=bool(observation and observation.get('healthy') and observation.get('progress',{}).get('at_batches',0)>0)
controller_started=bool((HERE/TAG/'launch.json').is_file() and observation and observation.get('healthy'))
stage='FALLBACK_ORIGINAL_PROTOCOL_STARTED' if started else ('NEW_AT_CONTROLLER_STARTED_INITIALIZING' if controller_started else 'CURRENT_GPU_GATES_PASSED_WAITING_SINGLE_SCIENTIFIC_REVIEW')
samples=[]
for p in sorted((HERE/'probes').glob('*/summary.json')):
    v=load(p)
    if v.get('CUDA_performance_measured') and not v['timing']['profile_instrumented']:
        samples.append(dict(label=p.parent.name,implementation=v['implementation'],batch=v['native_batch'],images_per_second=v['timing']['images_per_second'],seconds_per_batch=v['timing']['seconds_per_batch_mean'],peak_allocated_MiB=v['cuda_peak_allocated_MiB']))
table='| 实现 | Batch | 原始 images/s 范围 | 原始秒/batch范围 | 分配显存峰值 GiB |\n|---|---:|---:|---:|---:|\n'
for impl,batch in [('original',36),('02',36),('03',36),('original',72)]:
    rows=[v for v in samples if (v['implementation'],v['batch'])==(impl,batch)]
    if rows:table+=f"| {impl} | {batch} | {min(v['images_per_second'] for v in rows):.3f}–{max(v['images_per_second'] for v in rows):.3f} | {min(v['seconds_per_batch'] for v in rows):.3f}–{max(v['seconds_per_batch'] for v in rows):.3f} | {max(v['peak_allocated_MiB'] for v in rows)/1024:.3f} |\n"
for batch in (48,64):table+=f'| original | {batch} | NOT RUN | NOT RUN | null |\n'
text=f'''# S4 AT 直接终止、CUDA比较与正式重启回执

状态：{stage}。更新时间 UTC：{now}。

## 旧实验终止与保留

旧 RunID `s4_at_serial_seed0_20261029_v2` 已于2026-10-30T05:23:23Z通过本人专属 RRCTL 终止，真实终态为 ABORTED。终止前epoch87，86个完整epoch，39666 batches、20265次SGD、12次合法AMP跳过。epoch86完整checkpoint SHA256为 `925dcaf995a6880d987e218b1e28a72c6008a93bbcf0cb26e10001167ea8e3ee`，日志、失败、锁、终态全部保留。旧进程组及CUDA上下文已消失；另外三臂当时未启动。旧checkpoint不作为新训练输入。

精确终止证据：`execution/direct_stop_20261030/20261030T052322Z_OLD_AT_TERMINATED.json`。单次控制安全审查负面LIFECYCLE-01与主审修复闭环均保留；没有第二次控制审查。新队列CPU真实控制验证通过：严格串行、重复控制器拒绝、旧全局锁SHA不变。

## 实际CUDA优化证据

Candidate02/03各完成CleanAT4零容差组件：每侧16个真实native outer、10次SGD、6次合法AMP跳过；每一步PGD梯度/TAL/loss/扰动及外层模型、optimizer、AMP、BN、EMA/RNG哈希一致。它们未完成所有四臂与resume全套候选验收，且没有被采用。

已完成8个实现ABBA测速样本与4个Original Batch36/72 ABBA样本。下面是实际原始样本，不能当作通过资源可比性验收的加速排名。

{table}

Candidate02/03 ABBA均为INCONCLUSIVE：原锁定时钟、CPU负载、热/功耗限频占比条件未通过。Batch72 ABBA同为INCONCLUSIVE：时钟波动及ROI遥测不足。validated speedup百分比为null。首次bo1实际遭遇另一个GPU PID，已作废并保留；未操作该进程。没有放宽容差或将batch翻倍当作吞吐翻倍。

Batch48、64及CUDA事件分解未运行；限定派发窗口已过，停止扩展搜索以执行用户明确授权的回退。不能据此宣称它们不可行或更慢。没有可靠的全候选最优排名；最终采用通过原科学协议验收的Original Batch36，不采用Candidate或新Batch。PGD10、4/8预算、步长1/2、640、workers16、200epoch、nbs64、native SGD/AMP及完整GT保持原协议。新Batch的accumulation/BN/warmup/update/EMA差异仅作为未采用方案披露，实际原协议weight decay为0.0005625000000000001。

## 冻结当前源码预检与科学门槛

运行源码：`{SOURCE}`，已正常推送并读回远端SHA。19个科学文件与60个OriginalS4来源字节一致51ce已验收版本。GT/数据复用已接受清单，未重跑COCO、S4生成或全量导出审计。

当前源码真实四臂worker CUDA预检通过26个native outer，各臂均发生SGD，跳过数依次6/6/5/5，loss及状态有限；四臂新随机初始化SHA `{main['initial_sha256']}` 一致。独立DetectionModel基线与实际worker loss均453.893798828125，diff0/tolerance0。当前源码61个outer的完整epoch安全相位恢复通过，model/optimizer/scaler/scheduler/EMA/gradients/PGD RNG及attempt相位零差；不声称workers16预取轨迹逐位恢复。新正式训练仍从epoch0开始。

预检RunID：`s4_at_selected_native_20261030_gpu01`、`s4_at_selected_baseline_20261030_gpu01`、`s4_at_selected_phase_20261030_gpu01`。均完成exit0、completion health、wait与小文件pull一次；无并发本人GPU。Git读取发生两次TLS EOF，保留连接失败，并通过已连接GitHub官方API准确读回同一分支SHA后继续；未修改TLS/VPN/代理。

唯一新科学审查：{review['result']+' / '+review['decision'] if review else 'RUNNING / verdict not yet available'}。原始JSON与Markdown保留，正式启动还须真实科学闭环与不可变门槛上传。没有把CPU验证或源代码相同当作GPU结果。

## 新正式四臂状态

新控制器RunID：`{TAG}`。控制器是否已实际启动：{controller_started}；是否已记录正式训练batch：{started}。顺序固定 CleanAT4 → PoisonAT4 → CleanAT8 → PoisonAT8；均为新随机YAML seed0、200epoch、Original Batch36。其余三臂等待同一控制器串行调度，不同时启动。

当前进度：`{json.dumps(observation.get('progress') if observation else None,ensure_ascii=False)}`。
当前身份：`{json.dumps(observation.get('identities') if observation else None,ensure_ascii=False)}`。
日志：`/project/guoqinglong/projects/s4_at_robustness_20261027/outputs/{TAG}/result/S4AT4_Clean_seed0.console.log`。

新RunSpec/READY/launch及门槛见 `execution/direct_stop_20261030/{TAG}/`、`SELECTED_RUNTIME.json`、`SELECTED_GATE_UPLOAD_VERIFIED.json`；仅在实际文件存在时代表门槛已闭合。旧全局锁/终态未清理，采用经CPU验证的新独占flock与全新输出根。当前GPU若新CleanAT4已启动，则由其使用。

## 后续与结果边界

保持同一个ACTIVE监控器，禁止重复控制器、并发测试、热替换或旧checkpoint续训。四臂真实200epoch后核验连续epoch、last.pt@200、完整状态/optimizer/scaler/EMA、初始化/source及验收回执，复用各臂已计算的完整cleanVOC4952/all20 last主/best次评价，完成匹配预算全类别P/U/S/WCD/TCD3与最终发布。目前没有新的正式AT效果指标，不作性能或因果效应结论。COCO已接受并发布，保持原样。
'''
(DOC/'S4_AT_DIRECT_STOP_CUDA_OPTIMIZATION_AND_RESTART_REPORT.md').write_text(text,encoding='utf8')
(HERE/'ACTUAL_CUDA_COMPARISON_TABLE.json').write_text(json.dumps(dict(raw_samples=samples,validated_speedup_percent=None,selected='OriginalBatch36 fallback',batch48='NOT_RUN',batch64='NOT_RUN'),indent=2),encoding='utf8')
s.update(UTC=now,state=stage,candidate_runtime_commit=SOURCE,active_CUDA_probe=None,current_source_CUDA_preflight='PASSED_NATIVE26_BASELINE7_PHASE61',
    selected_batch=36,selected_implementation='original',new_protocol_scientific_review=review['result'] if review else 'RUNNING_SINGLE_REVIEW',
    active_formal_controllers=1 if controller_started else 0,AT_server_controller_started=controller_started,new_formal_controller_started=controller_started,
    formal_run_allowed=controller_started,formal_RunID=TAG if controller_started else None,current_progress=observation.get('progress') if observation else None,
    current_identities=observation.get('identities') if observation else None,current_source_CUDA_probe_paths=[str(HERE/'current_probes'/tag/'summary.json') for tag in ['s4_at_selected_native_20261030_gpu01','s4_at_selected_baseline_20261030_gpu01','s4_at_selected_phase_20261030_gpu01']])
s.update(CUDA_preflight='CURRENT_SOURCE_NATIVE26_BASELINE7_PHASE61_PASSED',scientific_review=review['result'] if review else 'RUNNING_SINGLE_SCIENTIFIC_REVIEW',
    formal_preflight='ACCEPTED' if controller_started else 'PENDING_SINGLE_REVIEW_AND_GATE',source_publication_API_verified=True,
    pending=['Complete four serial fresh200epoch arms and actual clean4952 evaluation/acceptance/publication'] if controller_started else ['Finish existing single scientific review','Main closure and immutable gate','Launch one fresh four-arm controller'],
    next='Inspect only the new formal binding and bounded progress; no more optimization probes' if controller_started else 'Await existing single reviewer; no duplicate review or GPU tests')
(DOC/'CURRENT_STATE.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(state=stage,formal_started=started,review=review['result'] if review else None,source=SOURCE)))
