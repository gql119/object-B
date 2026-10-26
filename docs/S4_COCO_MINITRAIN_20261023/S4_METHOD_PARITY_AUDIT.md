# Method parity audit — risk extension candidate

Preserved: original8/4/0,Qt/W/eta.85,TSFL39/VOC24 selection rule,SafeBG rank8/alpha.5/top3/K8/norm restoration,P3 coordinates,all original losses/weights,Frozen/Adaptive/Fourier/EOT/Adam/schedule,full25K/COCO80/originalval5K. No native Ultralytics TAL changes or framework-method source changes.

Declared difference: for a valid GT lacking native and independent native supervision, the NT clean-input sensitivity uses an explicitly GT-conditioned independent detection-risk loss. Its hard GT classification targets/uniform candidate weights are new in that previously undefined branch. Therefore global equality to the original strict estimator is NOT claimed.

Normal covered calls delegate unchanged; actual output/gradient equivalence remains to be measured. Instance-local cache dependencies isolate this branch without changing originalQ/map/cache code objects or module globals. Cache identity includes new estimator/adapter/spec SHA. Full preventive source/consumer checks and actual production sinks remain pending.


## Latest bounded acceptance and population validation
# COCO risk boundary extension: bounded validation accepted, population audit RUNNING

This is a Level C methodological boundary extension candidate, not globally equivalent strict Original S4. No formal Fourier24/poison25K/victim/evaluation was restarted. The old12step run and all failed validations remain immutable.

The valid small backpack has zero native TAL supervision even after same-TAL rematching; official labels and coordinates are correct. The new definition affects only that undefined boundary. Normal covered calls delegate unchanged. GT-conditioned candidates are explicitly not native positives; the original CIoU/BCE79/DFL gains, per-GT once/per-class mean/map normalization/equal-class mean remain. Qt, W=1-.15Rnt(1-Qt),8/4/0 and all other S4 mechanics stay fixed.

| Gate | Actual evidence |
|---|---|
| Covered COCO/VOC maps, records, full input gradients | CPU/CUDA exact, tolerance0; GPU comparison has original-repeat controls |
| Mixed image defined NTGT | All13 original per-object losses and full gradients exact; original14th undefined term stays null |
| Class accounting/normalization | All14 NTGT once; class-normalized equal mean exactly matches native clean-map cache |
| Boundary samples | Actual noPerson image25/2GT/1forward; Person-only29020/1GT/0forwards |
| Original VOC regressions |55 tests,0 failures,0 errors,0 skips; randomVOC fixture retained |
| Real Fourier/TFSS/Frozen/Adaptive/export | CPU/CUDA each15nativeforwards/EOT2/one isolated Adam update, finite directBG/rel coefficient gradients, actual weighted input sinks, native final-image/export exact |
| Decoded single-image budget | max2 bytes, overlap1 byte, outside0, within unchanged8/4/0 |
| One independent review |3524258 received negative F01; original verdict retained; no second/closing reviewer |
| F01 repair |6f54647 promotes GT endpoints before float64 center/size; actual helper counterexample correctindex1, true tie lowest index, invalid boxes rejected |
| Current F01 CPU/CUDA production validation | Both8-image/11nativeforward probes pass; real five-candidate case unchanged;155 source/7native/Teacher hashes independently audited CPU-only |
| Complete25K trigger/class/fraction and conditional-W census | RUNNING; no completed population results or prevalence estimate yet |
| Formal optimization | NOT AUTHORIZED for this Level C validation scope; NOT STARTED |

Default CUDA VOC ref-ref differences around1.8e-15 caused the first zero-tolerance harness failure. The common deterministic comparison domain is exact; production defaults are restored and no tolerance is relaxed. MixedCPU01 and02 remain failed overall: missing integration fixtures caused2 skips, then the retained-temp fixture lacked the cleanup API required by a lazy Torch import. The separate corrected regression run passes55 with no skips. The passed pre-regression mixed stage is labelled partial, not an overall pass.

The independent reviewer identified a real nearest-anchor false-tie bug, fixed in6f54647243e2a5c7bffe3b91ca2f734454fa30ab. Main closed F01 using current CPU/CUDA actual-helper and production-path evidence. The permitted action after closure is only the full25K train-only incidence/affected-eligible-Person conditional-W audit, with zero optimizer updates. Original undefined Rnt/W are null, never compared with imputed zero. All79 NT-class counters are retained; conditional W must not be called a global training distribution.

Actual active census: coco_dim_20261027_risk_population01, immutable source6f54647243e2a5c7bffe3b91ca2f734454fa30ab. One RRCTL finite workload, no new controller/monitor. An initial receipt upload failed before any RunSpec/job because the receipt parent was absent; this engineering failure was preserved, then a verified owned directory was created and exact SHA upload/readback passed. The actual workload is now healthy/running. The archived start observation is transient, not a completed result.

See RISK_BOUNDED_ACCEPTANCE_20261027 for the original negative verdict, main closure, actual independent audits, zero-skip regression summary and exact raw lean packet. Packet raw SHA18498ca8... differs from canonical-JSON/readiness SHAed1451e4...; both were verified, no content drift. No weights, images, original labels, profiles or secrets are included.


## Latest: complete25K independent acceptance

# COCO80 无 TAL 监督 GT：隔离验证最终报告

本次授权范围内的方案、实现、真实 CPU/CUDA 复验、一次独立审查及完整25K统计已经完成并通过主流程验收。**正式 Fourier 优化没有重新启动。** 当前补充分支属于 Level C 方法边界定义扩展，不能称为与严格 Original S4 完全等价的普通修复。

## 根因

确实是合法小目标缺少 TAL 有效监督。图像000000247692的 annotation1827724 是 backpack，官方 category27 对应 YOLO class24；GT13在640输入中的宽高约4.43×6.40像素。官方 JSON、全部23条标签转换、LetterBox与类别映射均核对正确。原生全GT TAL及独立同TAL重匹配都没有有效监督；5个几何候选的未截断 CIoU 均为负，float64也不能恢复正样本。这不是坐标错误、GPU占用或显存问题。

## 补充的数学定义

正常有效 TAL GT继续使用原函数、原软监督权重与损失。只有原生TAL和独立同TAL均无正的有效监督时，才定义独立风险候选集合 A_g：首先采用固定原生几何规则；若集合为空，按真实GT尺寸归一化的中心距离选择最近原生位置，完全相等时取原生索引较小者。端点先转float64，再求中心、尺寸与距离；不加epsilon、不改GT，不把候选称为原生正样本。

\[
\ell_{fb}(g)=\frac1{|A_g|}\sum_{i\in A_g}\left[
\lambda_{box}(1-\operatorname{CIoU}(\hat b_i,b_g))+
\lambda_{cls}\sum_{c\ne Person}\operatorname{BCEWithLogits}(z_{ic},\mathbf1[c=c_g])+
\lambda_{dfl}\operatorname{DFL}(\hat d_i,\operatorname{bbox2dist}(a_i,b_g))\right].
\]

保留原生gains=7.5/0.5/1.5。每个GT只计一次，先按原规则在类内平均GT损失，再求干净输入梯度范数，逐类做原归一化，最后等权平均当前图像出现的非目标类别。原始完整风险在未定义图像上仍记录null，不能用零替代来构造“新旧差值”。

改变的是 R_nt 在原规则无定义时的边界定义；Q_t、W_U=1−0.15R_nt(1−Q_t)、eta=0.85、8/4/0、StrictTSFL39/VOC24、Safe-BG、独立P3、全部原损失/Fourier/Frozen/Adaptive及数据和评价协议不变。原Teacher、完整25K校准、39通道、K8、79类基和P3原文件经Hash核对复用，未重训或重校准。

## 数值与实现验收

- 正常COCO及VOC20的原/新输出、记录和完整输入梯度在实际CPU/CUDA对照中完全一致，容差0。CUDA使用相同可重复比较环境并恢复生产默认设置。
- 混合失败图像中13个原本有监督的非目标GT，其逐目标损失及完整输入梯度完全一致；第14个原始未定义GT仍为null。14个GT均只计一次，逐类归一化和平均与实际原生缓存一致。
- 真实无Person与仅Person图像、多目标/重叠框、非法框与最近候选规则通过。55项VOC相关回归：0失败、0错误、0跳过。
- 实际CPU/CUDA Fourier全链路各执行EOT2、15次原生前向和1次隔离Adam更新，BG/rel梯度能回传到Fourier系数；TFSS确实进入Frozen/EOT/LetterBox与Adaptive。保存参数和原生最终生成/导出逐元素一致，单图PNG最大2字节、重叠1字节、框外0，满足原8/4/0。
- 恰好一次独立审查发现F01：先float32求中心再转double可能造成假平局。原否定结论未改；6f54647修复后，实际CPU/CUDA反例与生产路径通过，主流程关闭F01，没有第二或closing审查。

## 完整25K实际统计

| 项目 | 实测 |
|---|---:|
| 训练图像 |25,000|
| 有效训练GT / 非目标GT |181,475 /126,379|
| 保留的Person支持图像 |13,630|
| 触发补充的训练图像 |239（0.956%）|
| 触发补充的非目标GT |304（0.240546%）|
| 受影响的Person支持图像 |204（支持图像的1.496698%）|
| 上述204图像中纳入实际类归约风险梯度的补充GT |255|
| 无Person图像中的补充GT |49，分布于35张图像|
| 出现补充触发的NT类别 |43；全部79类分母与零计数均保留|
| 使用原生几何候选 / 最近位置补充 |304 /0|

255表示GT实际纳入了已验证的类归约梯度，不表示对每一个GT都单独测量了非零导数。无Person图像仅核对候选可用性，保持目标框外零扰动，不冒充TFSS活跃图像。

204张受影响支持图像中的8,749,487个Person域像素，条件W分布：min=0.85000002384、max=1、像素加权mean=0.97760065298。它是**受影响支持图像的条件分布**，不是全训练集W分布。原始R/W在这些图像上无定义，故不报告虚构的原/新差值；全训练集W分布仍为null。

独立CPU只读审计重新核对25K全部ID顺序及GT数、官方JSON的79类分母、13,630个支持ID和204份缓存Hash/策略身份；从已保存Q/R重新构建W，其每图均值、分位数、整体直方图及统计量全部与原GPU计算精确一致。没有再次扫描原图或运行模型。逐类别触发次数与比例见 COCO_RISK_TRIGGER_FREQUENCIES_20261027.json。

## 版本、保全与恢复结论

协议：coco80-ntgt-native-tal-then-gt-risk-v1。实际完成统计的不可变runtime为6f54647243e2a5c7bffe3b91ca2f734454fa30ab，RunID=coco_dim_20261027_risk_population01，25,541次实测模型前向，0次参数更新；它是验证统计作业，不是24epoch训练。代码与小报告在独立分支codex/s4-coco-risk-fallback-20261027发布；归档提交与实际runtime分别记录。

原始失败、12步Fourier checkpoint/Adam/scheduler/RNG、默认CUDA比较失败、两次旧回归夹具失败及原科学否定结论均保留。没有复用失败RunID、恢复旧12步、修改历史接受状态，或启动新监控/并发GPU调度器。

**是否满足正式重启条件：当前不满足。** 本次明确授权只含独立方案和验证，正式guard仍为false。隔离候选已经通过上述验收；若后续采用它，应明确作为带R_nt边界扩展的独立COCO协议，完成新的正式RunID/源配置绑定及正式批次预检，从原Fourier初始化开始。不能将它静默写成严格Original S4，也不能从旧12步Checkpoint续训。没有新增AP、LPIPS或方法有效性结论。
