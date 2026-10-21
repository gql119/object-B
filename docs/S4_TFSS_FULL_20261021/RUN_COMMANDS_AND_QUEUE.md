# 独立生产入口与队列准入

这些是未来RRCTL RunSpec中的workload命令；当前未提交远程launch，不使用raw SSH。部署时RRCTL必须固定本分支已提交的实际Git SHA和独立repo/output路径。已有RunID/binding只能inspect，不能重复launch。

本地CPU验证：
```powershell
python -B scripts/check_s4_tfss_full_cpu.py
```

工作负载配置参数：
```text
LOCK=docs/S4_TFSS_FULL_20261021/S4_TFSS_FULL_LOCK.json
RUN_ID=s4_tfss_full_overlap_seed0_20261021_v1
```

后续GPU smoke（唯一新输出、仅两个实际noise Adam更新，禁用正式指标和ingest）：
```text
python -B scripts/run_s4_tfss_full.py --phase smoke --lock docs/S4_TFSS_FULL_20261021/S4_TFSS_FULL_LOCK.json --output <locked_remote_root>/outputs/s4_tfss_full_overlap_gpu_smoke_seed0_20261021_v1 --admission <locked_remote_root>/receipts/QUEUE_ADMISSION.json
```

GPU smoke通过→对该commit进行一次独立科学审查→主代理逐条关闭实际blocker。正式生产命令：
```text
python -B scripts/run_s4_tfss_full.py --phase formal --lock docs/S4_TFSS_FULL_20261021/S4_TFSS_FULL_LOCK.json --output <locked_remote_root>/outputs/s4_tfss_full_overlap_seed0_20261021_v1 --admission <locked_remote_root>/receipts/QUEUE_ADMISSION.json --review <locked_remote_root>/receipts/PRERUN_CLOSURE.json
```

生产入口串行执行：原S4身份核验→相同初始化的24epochs/9144Adam新Fourier优化→新VOC数据导出及完整8/0审计→fresh random-init YOLOv8n seed0/200/batch36/workers16/AMP/nativeSGD→last@200主评价和best次评价→完整VOC20/PersonRecall/PUS/WCD/TCD3/空间诊断及对照报告。原生optimizer_step没有覆盖；AMP合法skip/backoff沿已修复S4路径。

已完成200轮而尚未开始评价的首次独立评价入口（不启动训练，也不覆盖已有评价）：
```text
python -B scripts/evaluate_s4_tfss_full.py --output <same_bound_run_output> --lock docs/S4_TFSS_FULL_20261021/S4_TFSS_FULL_LOCK.json --admission <fresh_bound_admission>
```

已完成正式评价的纯CPU报告重建：
```text
python -B scripts/s4_full_report.py --output <same_bound_run_output> --lock docs/S4_TFSS_FULL_20261021/S4_TFSS_FULL_LOCK.json
```
后两条仍要求训练/source/config/checkpoint/200行CSV和数据审计身份，不能把其它实验的AP混入Full。

## 不可跳过的队列门槛

新入口直接读取旧服务器pipeline_state.json，要求匹配原唯一recovery01 RunID、state=COMPLETED_AND_ACCEPTED、全部10个阶段ACCEPTED且active_job为空。还须有15分钟内的RRCTL队列审查receipt，绑定当前source/config和旧pipeline状态SHA，明确已查后续计划、queue_conflicts=[]、本用户活跃training root=0。

之后才检查原容量/I-O门槛（RAM32GiB、GPUfree24000MiB、disk150GiB、三窗口I/Owait约束）；CPU忙、GPU利用率和其他用户计算存在依旧是速度遥测，不据此抢占、暂停或调整他人进程。GPU空闲也不能绕过流水线未验收这一条件。

当前查询结果见QUEUE_STATUS.json。最近成功快照仍有MeanF/MedianF/DataAug/最终验收计划，因此没有任何新GPU任务被启动，也未改变旧server-owned调度。不会创建第二个监控。新Full任务的条件授权保存在USER_TASK_SOURCE.md；未来满足条件时无需再次询问同一授权。
