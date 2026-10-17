# Current Conclusions

### C001

<!-- research-memory:cef30d6773e30b012395b272:C001 -->
Type: finding
Status: SUPPORTED
Scope: experiment.S1.parent_selection
Effective at: 2026-10-15T01:06:21.640315+00:00

完整S1 primary targetAP50比S0升高6.267705pp，NT升高1.287801pp；按用户1pp双门槛锁S0作为CNLP/CUP父版本。不是统计显著性结论。

Source: E8d2242e8aa3ecddeee66871b
Source role: assistant
Source ref: research_workspace/analysis/s2_s3_counterfactual_20261014.md
Occurred at: 2026-10-15T01:06:21.640315+00:00
Received at: 2026-10-15T01:06:21.642310+00:00
Evidence: research_workspace/analysis/S1_PARENT_SELECTION_EVIDENCE.json, research_workspace/analysis/s2_s3_counterfactual_20261014.md
Protocol: matched clean VOC20 random-init YOLOv8n seed0 last.pt@200

### C002

<!-- research-memory:ee979e62edbf4f163c815923:C002 -->
Type: finding
Status: SUPPORTED
Scope: experiment.S2S3.scientific_admission
Effective at: 2026-10-15T02:45:40.097192+00:00

仅实现准入：当前0175916 CNLP/CUP配对higher-order梯度、原Parent等价、共享NTGT覆盖、CUP实际sink和checkpoint恢复通过；唯一独立审查scientifically_correct/allow_run，未验证最终AP效果或收敛。

Source: E026aff337122fa310df4fd52
Source role: document
Source ref: research_workspace/STATE.md
Occurred at: 2026-10-15T02:45:40.097192+00:00
Received at: 2026-10-15T02:46:04.139288+00:00
Evidence: research_workspace/analysis/S2_CNLP_ADMISSION_AND_LAUNCH_EVIDENCE.json
Protocol: fixed train-only calibration and bounded GPU production admission; no official metric

### C003

<!-- research-memory:ee979e62edbf4f163c815923:C003 -->
Type: execution
Status: OBSERVED
Scope: experiment.S2.formal_launch
Effective at: 2026-10-15T02:45:40.097192+00:00

S2唯一RunID s2_cnlp_s0_eps8_person_seed0_20261015_v1正式启动，首次真实状态noise_generation epoch1/5of9144steps。S3等待完整S2归档，不能重复launch。

Source: E026aff337122fa310df4fd52
Source role: document
Source ref: research_workspace/STATE.md
Occurred at: 2026-10-15T02:45:40.097192+00:00
Received at: 2026-10-15T02:46:04.139288+00:00
Evidence: research_workspace/analysis/S2_CNLP_ADMISSION_AND_LAUNCH_EVIDENCE.json
Protocol: source0175916; ParentS0; independent seed0; noise24epochs/9144steps and victim200 locked, not completed

### C004

<!-- research-memory:471e0b994be946c741a94f01:C004 -->
Type: finding
Status: MIXED
Scope: experiment.S2S3.parallel_resource_test
Effective at: 2026-10-15T09:03:57.838183+00:00

当前环境并行短测仅完成3/15步即因明显慢速结束。内存可共存但未证明提速：S2前/并行/停止后3.3157/.74184/1.3138步每分钟；S3三步301.38秒，峰值13.27GiB，无OOM。停止后未完全恢复，系统CPU负载/非相同样本混杂，不能全归因GPU竞争或外推长程。保持既定正式串行，不自动重试测试。

Source: E817d19cd580c4b6d01e0786c
Source role: document
Source ref: research_workspace/analysis/S2_S3_PARALLEL_RESOURCE_TEST_20261015.md
Occurred at: 2026-10-15T09:03:57.838183+00:00
Received at: 2026-10-15T09:08:23.635973+00:00
Evidence: research_workspace/analysis/S2_S3_PARALLEL_RESOURCE_TEST_20261015.json, research_workspace/analysis/S2_S3_PARALLEL_RESOURCE_TEST_20261015.md
Protocol: bounded resource probe source0175916;INCOMPLETE_STOPPED_EARLY;no formalS3/victim;no method changes

### C005

<!-- research-memory:458232d4e9d77eea883e0696:C005 -->
Type: finding
Status: MIXED
Scope: experiment.S1.target_regression
Effective at: 2026-10-15T13:41:10.904197+00:00

S1已完成匹配主评价：Person AP50上升6.267705pp、NT AP50上升1.287801pp。末轮训练平均delta仅低0.5276%，Lret低24.6949%、Lrel高0.9574%，不支持仅以整体噪声减弱解释。S1非目标敏感度预算未约束目标攻击效用，并将原重叠幅度缩放变为局部硬裁剪；后者严格触顶像素的投影导数为零。关键攻击区域被裁剪/梯度受限是OPEN机制解释，缺少局部触顶、梯度份额和完整等强度对照，不能宣称主因或收敛。保留S0父选择，不修改运行实验。

Source: Ef677aa9edbf3edf158f4e9f7
Source role: document
Source ref: research_workspace/analysis/S1_TARGET_REGRESSION_REVIEW_20261015.md
Occurred at: 2026-10-15T13:41:10.904197+00:00
Received at: 2026-10-15T13:41:25.731173+00:00
Evidence: research_workspace/analysis/S1_TARGET_REGRESSION_REVIEW_20261015.md, research_workspace/analysis/S1_S0_EXISTING_DIAGNOSTICS_AUDIT_20261015.json
Protocol: matched clean VOC20 random-init YOLOv8n seed0 last.pt@200; existing noise CSV epoch24

### C006

<!-- research-memory:8945a7b717b6c940fad27ef6:C006 -->
Type: execution
Status: OBSERVED
Scope: experiment.S2.noise_completion
Effective at: 2026-10-15T22:28:48.239896+00:00

唯一正式S2完成24epoch9144Adamsteps，实际69902.03118577832秒；末checkpoint CPU只读核验step9144/scheduler24/完整optimizer RNG CNLP query state与身份，SHA83cf11efac04a85a4e476c497cd55a2af56d4bb4b15a38838374ad9984f4af40，Frozen/Adaptive unchanged。16551条materialization清单已写，现有流程pixel_label_audit；审计、victim200和评价尚未完成，不宣称收敛/有效，不启动S3或重跑已完成阶段。

Source: Eb2b99d38ae221884e76aa700
Source role: document
Source ref: research_workspace/analysis/S2_NOISE_COMPLETION_20261016.md
Occurred at: 2026-10-15T22:28:48.239896+00:00
Received at: 2026-10-15T22:29:15.573442+00:00
Evidence: research_workspace/analysis/S2_NOISE_COMPLETION_20261016.md

### C007

<!-- research-memory:d78922b65190dba75343dc20:C007 -->
Type: finding
Status: MIXED
Scope: experiment.S2.formal_result
Effective at: 2026-10-16T01:47:40.352027+00:00

S2正式noise24/9144、全16551train/4952cleanval审计、随机victim200、last主best次完整VOC20评价已验收归档。Person AP50=.04036846271358735、NT=.6891226224462725、NTAP75=.509452875723384；相对S0目标+1.137504pp、NT-.257547pp，未显示目标抑制/总体保留改善；PUS和WCD/TCD3重算一致，单seed不作显著性或因果结论。S3仍按原锁运行，不依据S2调参。

Source: Ed38977e6980c31815cb900ec
Source role: document
Source ref: research_workspace/STATE.md
Occurred at: 2026-10-16T01:47:40.352027+00:00
Received at: 2026-10-16T01:47:57.423155+00:00
Evidence: research_workspace/analysis/S2_FINAL_ACCEPTANCE_20261016.json, research_workspace/analysis/S3_FORMAL_LAUNCH_20261016.md
Protocol: matched clean VOC20 random-init YOLOv8n seed0 last.pt@200

### C008

<!-- research-memory:16695b41e6881e596b343961:C008 -->
Type: execution
Status: OBSERVED
Scope: experiment.S3.formal_execution
Effective at: 2026-10-16T01:47:40.835272+00:00

S2完整验收归档且GPU空闲后，S3唯一RunID s3_cnlp_cup_s0_eps8_person_seed0_20261015_v1于2026-10-16 09:42HK通过rrctl启动。09:47实际epoch1/19of9144steps、elapsed132.159368s、19条记录关键损失梯度finite/outside0、soleGPU9652796434MiB；source0175916及原配对锁未改，原Parent独立初始化、不接续S2噪声。尚未完成，不声称收敛或最终有效。

Source: Edf685d51fb83a4e61a12073c
Source role: document
Source ref: research_workspace/analysis/S3_FORMAL_LAUNCH_20261016.md
Occurred at: 2026-10-16T01:47:40.835272+00:00
Received at: 2026-10-16T01:47:56.937855+00:00
Evidence: research_workspace/analysis/S3_FORMAL_LAUNCH_20261016.md


## Complete matched S2/S3 findings — 2026-10-17

Type: finding; Status: SUPPORTED for the recorded single-seed comparison only. Scope: VOC20/last@200/original matched random-init protocol. Source: docs/s3_archive/LOCAL_ACCEPTANCE.json, docs/S2_S3_COMPARISON.csv, docs/S2_S3_VOC20_PER_CLASS.csv. S3personAP50=.03441381201183112, NT=.6893913255219821, NTAP75=.5102903490276389. S3 is closer to S0 than S2 on targetAP, but S0 still has lower targetAP and higher overallNTAP; no robust efficacy/superiority or causal mechanism is established. WCD improves whileTCD3 slightly worsens vsS0. NoAP-based tuning, no Parent reselection.

Type: execution; Status: OBSERVED. S3 COMPLETE/VERIFIED/ARCHIVED and exit0. TFSS fixed candidate/source/CPU contracts remain as prepared. First current resource preflight is NOT_RUN/WAIT_FOR_CPU_RESOURCE, not a scientific failure: GPU empty butload28.2139>24. No remote upload/ready/launch occurred. Existing monitor retries actual resources without changing gate/science; fullTFSS still needs future human authorization.
