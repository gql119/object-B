# S5 三路 5-step 门槛

状态：NOT_RUN，尚未采用 CUP-S4 damage 定义，尚无 S5 正式 config hash/checkpoint/log/AMP pass。Parent/teacher/dataset/source/seed/授权 SHA 见 `S5_SOURCE_AUDIT_PROVENANCE.json`。

待比较的实际独立计算：原 S4、S4+基准类别分解、S4+CUP-S4。固定训练开发集沿用原 TRAIN32 manifest（SHA256 `3becd3c5c9338dfb887da6163df8063a8ed44303dbacc4351f15e5ba376be5a1`），保持两独立 batch16、每路最多5个更新、EOT2、seed0、原 Fourier/Frozen/Adaptive 初始化。不能使用 validation AP 做选择。

必须逐 EOT/step 核验 forward/loss/Fourier gradient/Adaptive update/spatial Q-R-W/delta/budget/GT资格；基准类别聚合恢复原 S4；各类有效 GT/EMA/pi/w/entropy/缺席策略；逐类和保护梯度贡献；AMP原生路径；原始PNG预算与标签。局部 CPU 分解测试只证明代数，不能填写上述 PASS。

后续 full-review 必须先当前源码 production GPU smoke，后一次独立 S5 科学审查；旧 S4/S3 reviewer 不作为 S5 pass。正式 RunID 尚未建立；无正式输出目录，无重复实验。
