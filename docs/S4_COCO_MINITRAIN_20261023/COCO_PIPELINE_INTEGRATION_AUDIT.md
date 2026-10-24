# COCO Experiment A 生产入口核对

记录时间：2026-10-25（香港时间）。本记录是源码检查，不是 GPU smoke、科学审查或实验结果。

本次任务优先恢复本地 Kaggle ZIP 数据；数据恢复不会自动把旧 VOC 入口变成 COCO 生产入口。

| 入口或组件 | 实际检查 | 后续要求 |
|---|---|---|
| `coco_s4/data.py`、类别/子空间适配 | 既有 45 个 CPU 契约通过；Person=0、80 类、79 个非目标子空间；原始 8 个数学源和 2 个 Victim 组件锁定 | 保留原公式和源码身份；新数据派生统计须重新生成 |
| `ue_framework/stages/train_victim.py` | `_prepare_train_run_dir()` 包含已有目录的 `shutil.rmtree`；入口调用缓存删除并默认支持 resume | 不调用这个旧通用入口；新候选须以独立、碰撞即拒绝、无删除路径执行，不修改旧实验 |
| `scripts/run_isolated_victim.py` | `VOC_NAMES` 和若干目标/预算检查明确绑定 VOC20/Person14 | 不能直接用于 COCO80 计算或把其输出作为 COCO 正式结果 |
| `scripts/run_s4_tfss_pipeline.py` | `BOUNDARY`、旧教师/语义工件、数据计数16551/4952和目标14均固定为原 VOC 实验 | 保留历史源，不运行或替换其身份锁来假冒 COCO 流水线 |
| 本机 Ultralytics `8.4.150` 原生 Trainer | 实际源码第627–630行：`final_epoch` 会触发 `self.validate()`，即使 `args.val=False`；第674行还有 `self.final_eval()` | COCO Teacher 的 train-only 契约需要显式实现并实际验证其零验证前向；仅设置 `val=False` 不足。本机源码检查不能代替后续服务器实际版本/源码绑定 |

仍需完成的正式阶段包括：完整 val2017 5K 图像/标签、COCO 生产入口和源码/配置/数据绑定、每个新科学候选所需 CPU→有界实际 GPU smoke→单次独立科学实现审查、fresh 资源/无冲突准入，以及独立固定版本的官方 COCOeval 环境。LPIPS 尚未验证，不填造值。

原始 S4 Safe-Soft05 + TFSS-Utility、8/4/0 预算、StrictP5、Safe-BG、TFSS、Frozen/Adaptive、Fourier24 和冻结优化协议保持不变。未启动新 GPU 任务，未创建新的训练调度器，未调用新 reviewer，未重新运行已接受的 Cutout/CutMix/MixUp/Full。

服务器实际源码复核（2026-10-24T17:12:59Z）：通过既有RRCTL只读取得本人环境7个原生源/配置，未import GPU库、未写远程文件。Ultralytics 8.4.150、torch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 2.4.6。实际 engine/trainer.py SHA256 `2eda410e0bfb8b65cb46eb975c465c5b62b0f3a3dad5a54967d6e9e1fcabecc7`；同样确认第628–630行会在final_epoch调用validate，第973–987行的final_eval还会strip_optimizer并运行validator。新Teacher入口需要保留原生训练更新/调度/AMP，单独阻止验证loader和两处验证调用，保留last@200及完整训练状态；不能直接沿用旧VOC入口或从这一源码读取推断GPU通过。完整指纹见ACTUAL_TEACHER_NATIVE_SOURCE_AUDIT.json。

已实现组件：coco_s4.teacher.COCOTrainOnlyTeacher。两个CPU探针已实际覆盖原生训练loop、最终epoch/final_eval阻断、零val/EMA前向和保存后optimizer/scaler/无hook；源码SHA见TEACHER_CPU_VALIDATION_20261025.json。完整生产入口与绑定/单服务器串行workflow尚未实现，不能标全流水线ready；实际CUDA和单次科学审查仍待当前候选完成后执行。
