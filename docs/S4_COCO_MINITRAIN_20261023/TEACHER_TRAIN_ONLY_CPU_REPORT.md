# Teacher train-only 组件的当前 CPU 验证

新增 `coco_s4/teacher.py`，调用原始 S4 Victim 组件及原生 YOLOv8n/SGD/预处理，不修改原始8个数学源或2个Victim组件。Teacher 数据字典不暴露val路径；验证loader/validator设置拒绝执行，原生final_epoch与final_eval均不会运行验证或strip checkpoint。被动EMA审计hook仅在原生保存前移除、之后恢复，不写入保存的模型。

真实本地CPU探针：72张合成图像、batch36、两批、1次实际SGD更新、有限loss/模型；validation loader图像0、validator/EMA前向0。last checkpoint保留optimizer/scaler，加载后的EMA不含审计hook，已有输出碰撞拒绝。另5类非法数据路由与15项冻结超参变更全部拒绝；80类名字顺序及Person0与官方映射匹配，8原始数学源/2Victim组件SHA保持一致。命令为 `python -B coco_s4/validate_teacher_cpu.py` 后 `python -B coco_s4/validate_teacher_contracts.py`，全部产物保留。

当前本地CPU为Python3.12.14/Torch2.14.0+cpu，服务器为Python3.11/Torch2.7.1+cu128；CPU组件通过不能代表服务器环境/GPU、完整Fourier损失链、串行入口或科学审查通过。未运行新GPU smoke、未调用reviewer、未创建正式COCO绑定。先前CPU保存失败已保留，并通过移除/恢复被动hook的边界修复后重新验证；不重跑已接受的数据或旧实验。

新任务明确自动启动：在真实验收门通过后无需再询问。Teacher继续严格train-only，固定last@200后另做完整clean val2017的只读官方COCOeval，不用val更新参数/估计方法统计/选checkpoint或调参。下一步继续完整生产入口、依赖、source/config/data绑定与单服务器串行工作流实现，按CPU→当前候选实际GPU smoke→单次独立科学审查闭合再正式运行。完整实现仍为OPEN。
