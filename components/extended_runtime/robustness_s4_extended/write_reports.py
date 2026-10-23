"""Preparation reports from real audit and CPU receipts, never invented efficacy."""
import datetime,json
from pathlib import Path
from .common import ROOT,CONFIG,load,sha,atomic

def main():
    r=ROOT/'robustness_s4_extended/reports';r.mkdir(exist_ok=True)
    c=load(CONFIG);a=load(ROOT/'execution/SERVER_READONLY_AUDIT.json');t=load(r/'CPU_TEST_RESULT.json');refs=load(ROOT/'execution/REFERENCE_SOURCE_MANIFEST.json')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();noise=a['full_noise_status']
    def write(name,s):(r/name).write_text(s.strip()+'\n',encoding='utf-8')
    atomic(r/'REFERENCE_SOURCE_MANIFEST.json',refs)
    atomic(r/'AUDITED_SOURCE_IDENTITY.json',dict(UTC=now,original_S4_identities=a['identities'],original_S4_manifest_count=16551,samples=a['sample_checks'],full_source_rehash_now=False,prior_full_audit='old accepted pipeline; all16551+clean4952 previously independently accepted',versions=a['versions'],ultralytics_native_sources=a['ultralytics_sources'],installer=a['ultralytics_installer'],native_git_sha='UNKNOWN_distribution_has_no_direct_url; actual12sourceSHA256bound',predecessor=a['full_state'],predecessor_process=a['full_process'],locked_train_args=a['full_locked_train_args'],old_eight_defenses='actual server all10stages ACCEPTED / COMPLETED_AND_ACCEPTED',old_state_sha256=a['old_pipeline_state_sha256']))
    write('EXTENDED_ROBUSTNESS_AUDIT.md',f'''
# 新增 S4 鲁棒性：只读审计

审计记录生成于 {now}。实际远程用户 guoqinglong。没有修改旧流水线、进程、原始数据、标签、环境或 checkpoint。

原八项 JPEG50/JPEG30/Gray/BDR/Gauss/MeanF/MedianF/DataAug：读取实际服务器 recovery01 pipeline state，COMPLETED_AND_ACCEPTED，10个阶段均 ACCEPTED、active_job=None，状态 SHA256 `{a['old_pipeline_state_sha256']}`。不是仅依赖本地状态推断。它们的正式结果已在旧独立分支发布。

当前优先任务是 S4-TFSS-Full：`{c['predecessor']['run_id']}`；读取时阶段 `{a['full_state']['stage']}`，第{noise['epoch']}/24轮、{noise['steps']}/9144次更新，PID{a['full_process']['pid']}/startticks{a['full_process']['start_ticks']}。这是审计快照，不是预计最终进度。该任务的产物只作后继启动门槛，绝不作为新增 Poison 数据。

新增 Poison 唯一来源：`{c['baseline']['parent_noise_run_id']}`，原始 Safe-Soft05+TFSS-Utility。训练集16551、clean val4952、Person14/VOC20。Noise最终训练状态、数据完整性和物化清单 SHA 分别为 `{a['identities']['parent_noise_checkpoint_sha256']}`、`{a['identities']['parent_dataset_integrity_sha256']}`、`{a['identities']['parent_materialization_manifest_sha256']}`。本轮重读这三份产物及首尾两个确定样本/两组GT；没有遍历16551张图像。此前旧流水线的完整哈希审计保留；Full验收后、GPU前再做一次原S4全源核验。

Clean/S4 路径从已审计旧 lock 读取，见 configs/QUEUE.json 的 baseline。数据视图在新增输出目录创建图片/GT只读链接；YOLO缓存写在新视图，不写原始目录。训练变换在线执行；所有验证集均链接未经变换的原始 clean VOC。没有导出新的 S4 噪声。

实际 Victim：random YAML YOLOv8n20cls，seed0、epochs200、batch36、workers16、imgsz640、AMP True、SGD、lr0/lrf=.01/.01、momentum.937、weight_decay.0005、cos_lr=True、patience0、pretrained=False。原生 warmup3/.8/.1、HSV .015/.7/.4、translate.1、scale.5、flipLR.5、Mosaic1、close_mosaic100、CopyPaste0从已安装 default.yaml 的精确字节绑定。last@200主评价、best次评价。未发现与用户所列冻结协议冲突。新增三项都重训匹配 Clean，不复用旧八项分母。

已安装 Ultralytics8.4.150（pip；distribution无direct_url，不能声称某个上游Git SHA）。精确12份实际源文件SHA见 AUDITED_SOURCE_IDENTITY.json，包括检测训练入口、损失、TAL、增强、数据构建、指标和默认参数；torch2.7.1+cu128/torchvision.22.1+cu128/numpy2.4.6/Pillow12.3.0/OpenCV5.0.0.93。未升级或安装依赖。

当前 Full 是独立的 run_s4_tfss_full.py 串行入口，不存在可安全热插拔的旧后继接口。它的 native终态是 `complete`，没有原生 COMPLETED_AND_ACCEPTED 文件。因此新增独立 successor 在 native complete 后读取实际9144完成记录、原新数据清单/GT/val/完整性哈希、200CSV/embedded last、last/best及全指标、optimizer/scaler/scheduler/RNG、报告来源，形成自己目录的 PREDECESSOR_ACCEPTANCE.json。只有该适配器完整通过并写出 COMPLETED_AND_ACCEPTED，才放行；不修改 Full 状态、不以进程消失/GPU空闲/Finished替代验收。

服务器等待器只有标准库/只读小状态轮询（300秒），单实例 flock、原子状态持久化、原 Supervisor/PID-startticks/exit凭据复用、普通独立systemd用户scope；不改变OOM保护和优先级。失败不跳过、不缩batch、不重置RunID；已有job只能收养，不能重启。所有输入/输出/旧历史保留，无删除。
''')
    write('REFERENCE_IMPLEMENTATION_AUDIT.md','''
# 原论文、官方实现、检测适配的区分

论文：[MI-UE ICLR2026, §6.3 / Table5](https://arxiv.org/html/2603.03725v1#S6.SS3)，分别测试三种增强与OP；表中是CIFAR10分类准确率，不能移作VOC检测的AP或推荐最优参数。本项目不复现MI-UE毒化方法，只审计其防御参考。

MI-UE官方代码固定 commit `4cd387978c08b0c85eed9afc0b849d0b8777691c`，四份指定源文件和requirements原始字节已留在 execution/references。每文件永久链接和SHA见 REFERENCE_SOURCE_MANIFEST.json。

| 方法 | 官方分类实现 | 已安装Ultralytics实际行为 | 本项目检测协议 |
|---|---|---|---|
| Cutout | cutout.py默认n_holes1/length16；32x32上名义面积25%，中心均匀、边界裁剪、乘零mask。dataloaders.py的CIFAR评估分支实际调用默认构造（注释提到CIFAR100长度8但调用没有传8），无单独概率分支，等价每图尝试p1；不改单标签。RandomCrop32,padding4和Flip.5仍保留 | 无直接采用的Detect Cutout入口 | 在线 uint8图，n1、名义面积.25（宽高各.5）、p1、fill0。所有GT遮挡上限.5；任何类别超限拒绝整个候选，最多8次，之后跳过；不裁GT、不删除GT。真实面积因边界裁剪及拒绝而变小，不能声称等同CIFAR强度 |
| CutMix | evaluation_sl.py按epoch创建Kornia RandomCutMixV2(input,class)，分类label mixing经loss_mix使用；requirements固定Kornia.6.10；默认num_mix1/p1/same_on_batch=False，cut_size与beta传None，该版实际generator解析为cut_size[0,1]、Beta(1,1) | Detect CutMix实际存在：Beta1，3候选区；仅接受与所有主图GT零相交区域；来源目标bbox可见面积>=.1（Detect），裁剪到粘贴区，拼接整数cls/Instances；不存在分类CE混合 | 调用该native实现，唯一额外标注保护是donor可见面积>=.5且裁剪后边长>=2px。主图GT不遮挡。p1，Beta1，3候选。不满足则跳过。严重截断目标排除；允许的截断按原生规则裁框，和Cutout不裁框的策略不同 |
| MixUp | Kornia.6.10 RandomMixUpV2(input,class)，默认p1/same_on_batch=False，lambda_val=None在该版generator对应Uniform(0,1)；loss_mix按混合label计算分类损失 | Native Detect MixUp取Beta(32,32)，uint8加权图像，拼接双方所有Instances和class IDs，不使用分类lambda*CE或额外实例损失权重 | 原生Beta32/32、p1；保留整数GT实例，双方同一arm采样。分布与Kornia分类版本不同，明确是YOLO检测适配 |

所有新增方法互斥。原生Mosaic、HSV、flip、affine、CopyPaste0不叠加独立第二套。native close_mosaic100同时把MixUp/CutMix概率置零，因此它们仅前100epoch有效；Cutout在最终Format之前、HSV/flip之后执行全部200epoch。Clean/S4均采用同一调度；这不是强制200epoch一直MixUp的另一协议。

随机机制：保留原生DataLoader worker seed和基础增强RNG；新增操作以SHA256(seed0, worker读取到的epoch, image stem, operation)生成局部seed，保存/恢复Python/NumPy全局状态。记录每worker前4次增强的GT遮挡/样本seed。预取边界可能读到旧epoch；不声称任意worker调度下Clean/S4每个batch遮挡完全一致，不声称精确DataLoader续训。相同sample/epoch/op的新增随机机制可复现；基础图像内容/原生albumentations仍可能影响数据流。

Cutout的“可见”是bbox遮挡面积代理，不能从bbox保证实例分割级可见性。实际遮挡率与跳过率可能依赖GT几何，需结合配对Clean绝对AP和完整非目标指标解释。

OP官方仓库为NeurIPS2023 What Can We Learn from Unlearnable Datasets?，固定 commit `866d1dab6420bb1ea99796c8a263f3b0562e23e9`。Step1将固定尺寸未归一化图像展平，训练无bias线性分类器CE，默认20epochs/SGD lr.1/batch1024/MultiStep50%及75%；类方向是fc.weight各行。代码中可视化部分硬编码10类3x32x32，不能直接套VOC20。Step2从同一poison identifier加载该线性模型，取权重行，normalize并QR，project_data依次扣除正交向量分量，实质X-(XQ)Q^T；reshape成图像后仍有原分类强增强与Normalize（默认未关闭），clean test默认不投影，另有project-test选项。原始project_data不把值自动夹回[0,1]；不要把本项目候选重建clip声称为原版实现。OP不是S4的语义L_orth。
''')
    write('OP_DETECTION_ADAPTATION_REVIEW.md','''
# OP 检测适配：BLOCKED_PROTOCOL_APPROVAL

本报告是方案比较，不是OP正式实现或已验证防御效果。op_prototype.py只有CPU SVD有效秩/正交投影/有限范围重建数学函数，不导入数据集或训练器；worker CLI没有op选项。配置enabled=false、OP_PROTOCOL_APPROVED=false。任意旧S4 L_orth均不在此定义中。

| 方案 | 输入与监督 | 计算/存储 | 关键科学风险 |
|---|---|---|---|
| A Image-level multi-label | 固定resize/letterbox全图，20维多热标签；线性BCE，所有类权重估计全图子空间 | 640x640RGB D=1228800、20方向float32约93.75MiB；投影O(NDr)，绝不建立DxD矩阵 | 与原单标签CE不同；共现与背景捷径混入；全图统一坐标/resize改变小目标；投影可能抹去合法检测信息，包括非目标 |
| B Instance-level pooled projection | 每个训练GT ROI resize64x64、所属类CE；共享20类线性模型/所有类方向的一个公共基；每个原始ROI投影重建 | D=12288，20方向约.94MiB；成本O(Ninst*Dr)，额外线性训练按实例数，暂未扫描全数据统计Ninst | 小目标放大、ROI背景和resize往返；不能把20类方向当成纯毒化噪声；需要规定多实例覆盖/融合，保证单次同时重建与像素范围 |
| C Class-conditional ROI | 各类或各实例条件基/掩码，按所属类别作用局部 | 多个basis/cache，至少约20倍B方向存储（依秩设计）；重叠像素融合成本 | Person/NT框覆盖同像素；顺序投影不交换，后一个可再次引入前一个分量；多类别强度不等/语义误删；GT相关处理可能使比较失公平 |

推荐讨论 **B的共享class-agnostic pooled basis**：每类均参与线性方向估计，不能只为Person训练；同一基作用所有类别ROI。所有ROI先从同一原始图像提取并独立重建，回贴时按每像素覆盖数等权平均，最后一次clip/round，避免顺序覆盖。背景ROI之外不变；重叠区需单列像素统计/各类受损情况。这包含ROI空间复原的明确额外假设，不能称为原论文标准检测版本，也不同于C的按类选基。

仍需用户确认：选择A/B/C、ROI分辨率、原图到ROI插值/坐标、投影模型输入范围/监督/训练轮数、线性优化器/方向有效秩、是否center、clip策略、ROI重叠融合、训练样本来源、Clean和S4分别估计投影器还是共享、额外训练预算和与原论文公平比较方式。推荐若采用B，应分别在各arm自己的原训练数据训练同协议投影器，不能用clean validation拟合方向或调参。20epoch只是原分类默认的预算参考，不是获批检测设置。

额外开销未实测：三项增强已有6次200epoch，OP若获批还需2个投影器+2个200epoch Victim以及全数据投影；D和Ninst主导成本。绝不现在训练投影器或全量估计。若正常语义/非目标AP受损显著，必须原样报告；不能只看Person恢复和S。

CPU原型使用SVD有效秩替代原代码未做rank检查的QR，处理重复/零向量；这是数学原型差异。测试投影幂等性、与基正交、零秩、非正交拒绝、重建范围。尚未实现或验证真实检测ROI流水线。

未来正式OP需额外保存raw/reconstructed/clipped像素范围、RMS/有效秩/clip比例、同一冻结特征提取器的改变（不能把像素变化当作feature实测）、Person/NT全指标、GT重叠分组和所有非目标损伤。当前这些正式统计全部NOT_RUN。
''')
    write('EXTENDED_ROBUSTNESS_TEST_REPORT.md',f'''
# CPU验证与未完成门槛

实测时间 {t['UTC']}：{t['tests']}项、failure={t['failures']}、error={t['errors']}、skip={t['skipped']}；状态{t['status']}，Python AST检查{t['parsed_files']}文件。读取实际VOC图片0，GPU steps0；所有合成fixture保留，未删除checkpoint。

覆盖Cutout的单Person、Person+Dog、Person+Horse重叠、无Person、小目标、完全遮挡、边界、空标注；20类同策略、重叠孔洞面积并集、概率0、seed稳定、非法类/坐标拒绝。原生检测MixUp多实例/空GT与像素范围；CutMix不遮挡主图、严重截断排除、无合格donor跳过；真实YOLODataset+DataLoader合成CPU batch；实际native与概率0候选img/cls/box/batch_idx数值完全相等；close_mosaic100重建后CutMix/MixUp确实p0。还有OP投影数学、signed P/U/S和零分母、队列只读等待/身份拒绝、OP不可调用等契约。

CPU_VISUAL_AUDIT.png已人工视觉核验10组前后图与GT：Cutout拒绝时图像/框不变；允许时全框保留；CutMix只新增有效来源框；MixUp三框拼接；图像为合成输入，不是S4效果示例。

首次Windows CPU测试发现np.random.randint(2**32)默认int32越界，发生于Cutout随机种子生成；已显式使用uint64修复。失败原始结果在execution/CPU_TEST_FAILURE_V1.json/.txt保留。修复后全部测试通过（随后增加冻结协议漂移拒绝测试，共31项），没有修改算法、正式训练配置或资源阈值来规避失败。

本地torch2.14.0+cpu/numpy2.5.4与远程冻结版本不同，因此这些结果不证明远程CUDA数值等价。服务器再运行同一合成CPU契约和真实独立scope/单实例/已完成job收养检查；该凭据另存CPU_SERVER_SMOKE_VERIFIED。当前新增GPU Smoke=PENDING_GPU_SMOKE；唯一独立科学审查=PENDING_GPU_SMOKE_THEN_ONE_SCIENTIFIC_REVIEW；6次正式训练NOT_RUN；OP=BLOCKED_PROTOCOL_APPROVAL；无AP可填。

运行：`python -B -m robustness_s4_extended.validate_cpu`。测试结果JSON含确切版本、配置SHA和原始文本SHA。未运行全仓回归、性能benchmark、全量VOC遍历、模型预训练或OP方向估计。
''')
    write('RUN_COMMANDS_AND_QUEUE.md',f'''
# 后继队列、资源和状态

当前准备分支 codex/s4-robustness-extended-defenses-20261022；runtime源码SHA以实际RunSpec和run_ref为准，不用后来文档HEAD替代。新root只从旧锁定父目录派生，不触碰旧root：`{c['root']}`。RunID `{c['pipeline_run_id']}`；正式数据源仍为original S4，不是Full。

顺序：WAITING_PREVIOUS_PIPELINE → CHECK_PREVIOUS_ACCEPTANCE（自己的完整COMPLETED_AND_ACCEPTED收据）→ VALIDATE_NEW_CODE/source/native/data → 三个独立few-step GPU smoke（各<=42计算单元：Clean/S4各<=20真实batch +2个更新前loss forward；无正式validation/ingest）→ 一次科学审查凭据 → Cutout Clean200→S4 200→accept → CutMix Clean200→S4 200→accept → MixUp Clean200→S4 200→accept →部分报告归档→BLOCKED_PROTOCOL_APPROVAL。OP不阻塞前三项归档且不会被自动执行。

当前不新增GPU工作。server300s只读小状态轮询；网络中断不等于训练失败。单实例flock和job intent/PID-startticks/exit凭据拒绝重复；CPU监督smoke直接使用正式继承的run_job/systemd scope，无GPU/真实OOM注入。所有失败保留，不自动exact-resume、不重启原RunID，不更改epoch/batch/AMP/LR来绕过失败。

资源：同一GPU串行；沿用原三窗口capacity/I-O规则，RAM>=32GiB/GPUfree>=24000MiB/disk>=150GiB，持续全部3窗口IOwait>=20%阻塞新stage。CPU busy/load、GPU利用率/他人compute存在仅为速度遥测，不新增硬门槛。健康active own job不套空闲GPU门槛。无sudo、环境升级、进程干扰或原图写入。

资源预算（配置推算，未benchmark）：6次200epoch、每epoch ceil(16551/36)=460batch，共552000训练batch，另各last/best在4952clean val评价（共12次checkpoint评价），训练epoch自带原生validation亦保留。每次新增配对独立训练，无历史Clean分母。图像/GT使用只读链接，无6份全量图片；显存/CPU/RAM随冻结batch36/workers16/cacheTrue，不能根据猜测调整。保留last/best、每epoch安全副本/optimizer/RNG，磁盘开销需在运行准入确认；不自动清理。小时数尚未测，不能承诺完成时间。

只读状态命令（在本新工作树目录、使用既有RRCTL环境变量/解释器）：

`python -B -m robustness_s4_extended.control status --profiles ../s4_robustness_evaluation_20261019/execution/profiles.json`

执行器只通过 RRCTL Controller/ProfileStore/transport；新CPU控制run和后继run各唯一绑定。存在binding时仅status，不再launch。当前执行凭据和实际部署状态见execution以及reports/DEPLOYMENT_STATUS.json，不能把PREPARED描述成已部署。

审查门槛来自项目 `F:/autodl-tmp - 副本/ue_project/.agents/skills/pre-run-implementation-review/SKILL.md` 的 `isolated remote GPU few-step smoke -> one scientific implementation review -> official training/evaluation`。服务器能独立等待/训练/评价/顺序推进；科学审查凭据须由独立审查服务或现有唯一Codex监控签发，不能由调度器自证。服务不可用时保持PENDING_SCIENTIFIC_REVIEW，不需要本地持续连接维持server进程，但不会谎称无外部门槛。
''')
    write('USER_AUTHORIZATION.md','''
# 授权范围记录（2026-10-22本对话直接用户任务）

用户明确授权现在只读审计、参考官方源码、独立分支开发Cutout/CutMix/MixUp检测适配、隔离OP数学原型、无GPU测试、低资源后继调度准备和部署。只有正在运行的S4-TFSS-Full完整完成并通过实际完整验收，后继才可自动GPU smoke及配对200epoch；顺序Cutout→CutMix→MixUp。OP必须单独选择并批准检测协议；当前BLOCKED_PROTOCOL_APPROVAL。不得中断、修改或抢占Full；原S4数据/噪声/标签/方法冻结。六个新Victim需要新匹配Clean，clean VOC val、last200主best次、全VOC20/PUS/WCD/TCD3，source/config/data/checkpoint可追踪。不能进行AT/新种子/新S4扰动、环境升级、历史删除或其他用户干扰。

这是人类明确任务的执行范围摘要，不是新增科学结果，不替代原对话。一般代码开发与已满足条件的后继运行不重复问批准。OP独立审批及项目当前源GPU smoke→一次科学审查门槛仍保留。
''')
    print(json.dumps(dict(reports_written=7,CPU_status=t['status'],old_pipeline='COMPLETED_AND_ACCEPTED',Full_epoch=noise['epoch'],Full_steps=noise['steps'],new_GPU_steps=0)))
if __name__=='__main__':main()
