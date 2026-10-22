# 新增 S4 鲁棒性：只读审计

审计记录生成于 2026-10-22T03:37:50.291070+00:00。实际远程用户 guoqinglong。没有修改旧流水线、进程、原始数据、标签、环境或 checkpoint。

原八项 JPEG50/JPEG30/Gray/BDR/Gauss/MeanF/MedianF/DataAug：读取实际服务器 recovery01 pipeline state，COMPLETED_AND_ACCEPTED，10个阶段均 ACCEPTED、active_job=None，状态 SHA256 `53c2ba7167bcfbf17bd077f08299034c8b3db95029f7cf3223a29b94e7906590`。不是仅依赖本地状态推断。它们的正式结果已在旧独立分支发布。

当前优先任务是 S4-TFSS-Full：`s4_tfss_full_overlap_seed0_20261021_v1`；读取时阶段 `noise_generation`，第9/24轮、3129/9144次更新，PID2829581/startticks325195800。这是审计快照，不是预计最终进度。该任务的产物只作后继启动门槛，绝不作为新增 Poison 数据。

新增 Poison 唯一来源：`s4_tfss_utility_seed0_20261017_v1`，原始 Safe-Soft05+TFSS-Utility。训练集16551、clean val4952、Person14/VOC20。Noise最终训练状态、数据完整性和物化清单 SHA 分别为 `4413e45aa9c448bb768cac50e0ae40b5b0a1a580e38ff37cf66023bdab07c6e5`、`3eece1fc62df963f7d4aa15d378395273bd3963f812c0f7764ea9115dc82fe82`、`194d05cba8674c5699f83ad70178e93d6df7389a54fb0914d140342607598d36`。本轮重读这三份产物及首尾两个确定样本/两组GT；没有遍历16551张图像。此前旧流水线的完整哈希审计保留；Full验收后、GPU前再做一次原S4全源核验。

Clean/S4 路径从已审计旧 lock 读取，见 configs/QUEUE.json 的 baseline。数据视图在新增输出目录创建图片/GT只读链接；YOLO缓存写在新视图，不写原始目录。训练变换在线执行；所有验证集均链接未经变换的原始 clean VOC。没有导出新的 S4 噪声。

实际 Victim：random YAML YOLOv8n20cls，seed0、epochs200、batch36、workers16、imgsz640、AMP True、SGD、lr0/lrf=.01/.01、momentum.937、weight_decay.0005、cos_lr=True、patience0、pretrained=False。原生 warmup3/.8/.1、HSV .015/.7/.4、translate.1、scale.5、flipLR.5、Mosaic1、close_mosaic100、CopyPaste0从已安装 default.yaml 的精确字节绑定。last@200主评价、best次评价。未发现与用户所列冻结协议冲突。新增三项都重训匹配 Clean，不复用旧八项分母。

已安装 Ultralytics8.4.150（pip；distribution无direct_url，不能声称某个上游Git SHA）。精确12份实际源文件SHA见 AUDITED_SOURCE_IDENTITY.json，包括检测训练入口、损失、TAL、增强、数据构建、指标和默认参数；torch2.7.1+cu128/torchvision.22.1+cu128/numpy2.4.6/Pillow12.3.0/OpenCV5.0.0.93。未升级或安装依赖。

当前 Full 是独立的 run_s4_tfss_full.py 串行入口，不存在可安全热插拔的旧后继接口。它的 native终态是 `complete`，没有原生 COMPLETED_AND_ACCEPTED 文件。因此新增独立 successor 在 native complete 后读取实际9144完成记录、原新数据清单/GT/val/完整性哈希、200CSV/embedded last、last/best及全指标、optimizer/scaler/scheduler/RNG、报告来源，形成自己目录的 PREDECESSOR_ACCEPTANCE.json。只有该适配器完整通过并写出 COMPLETED_AND_ACCEPTED，才放行；不修改 Full 状态、不以进程消失/GPU空闲/Finished替代验收。

服务器等待器只有标准库/只读小状态轮询（300秒），单实例 flock、原子状态持久化、原 Supervisor/PID-startticks/exit凭据复用、普通独立systemd用户scope；不改变OOM保护和优先级。失败不跳过、不缩batch、不重置RunID；已有job只能收养，不能重启。所有输入/输出/旧历史保留，无删除。
