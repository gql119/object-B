# 历史防御与 Clean 复用

已核验平均3、Gaussian3sigma0、Median3、strong_aug的Clean均有200行CSV、complete状态、匹配随机YAML/initSHA/SGD/LR/scheduler/batch36/workers16/AMP及clean源目录，实际lastSHA与历史评价身份一致。四组不重训；进入PhaseB前使用同一cleanval重评既有last，补齐历史未保存AP75/AP50-95/Precision并核对原AP50/Recall。

旧通用robust_transforms Gaussian sigma1与正式历史sigma0不同，不复用其默认值。旧apply_robust_transform.py包含shutil.rmtree重置逻辑，本轮绝不调用。正式历史过滤在YOLO resize/augmentation之前；本轮离线PNG方案需与其过滤RGB像素全量一致。strong_aug只将degrees=10/shear=2替换原生值，其他原生增广不变，不重复叠加。PGD旧clean只完成121/200，不能作为AT正式对照。
