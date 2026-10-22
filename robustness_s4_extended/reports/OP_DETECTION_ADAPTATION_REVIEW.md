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
