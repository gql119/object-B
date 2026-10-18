# P3/P5 Stage A 源码与运行层审计

状态：静态源码确认；实际 Frozen 模块类型/张量形状将由 Stage B 当前权重运行记录核验，尚未填写 runtime PASS。最新授权 SHA256 `6eed7e9a239e171821ebe4c2db8ead3e1c623b933098401197ecbaef6c08c7a3`。Parent source/teacher/data/config/seed 元数据见 `S5_SOURCE_AUDIT_PROVENANCE.json`；S5 正式配置、checkpoint、AMP及victim结果未运行。

1. 正式 S0 的 strict-P5 Safe-Soft05 配置在 `B2SemanticTrainer.__init__` 同时指定 P5目标路径和P3保持/orth路径。S4通过原 `B2SemanticTrainer` 完全继承 Frozen L_ret；TFSS-U只在扰动合成末尾乘 W_U。45份原始文件仍不变。
2. `feature_layer=model.21` 与 `adaptive_feature_layer=model.15` 为已归档配置和运行身份。不得只凭编号判断语义：Stage B将从真实YOLO权重读取这两个模块类型和clean/poison输出形状，并验证分辨率4倍关系（输入640下对应stride8/32）。如果不同则失败保留证据，不能继续假设。
3. 原Frozen保持路径来自 `orth_clean/orth_poisoned=self.multi_features['model.15']`，将真实非目标GT矩形经原padding/letterbox/area-resize映射到该层，使用 `masked_prototype(min_pixels=1)`。全通道向量做 squared-L2-sum，跨有效实例mean，lambda_ret=.25；不使用TSFL目标24通道。
4. 原Frozen clean分支detach；poison分支经过同一个冻结模型和原扰动路径向Fourier传播。Frozen L_ret不会更新Adaptive模型参数。原Adaptive自身的一步虚拟更新与其NT保持项通过可微路径影响Fourier，本轮保持它不变；不能把这两个NT项混为一个重新加权。
5. TSFL的Person敏感24通道与Safe-BG发生在P5的目标BG/relative路径，作用与非目标保持不同。L_orth保留原P3路径、通道/子空间/权重；P5层探针仅替换Frozen L_ret归约，不能把 L_orth 一起切换。
6. P5对照使用该层全部通道和同一GT语义/原距离，不使用那24个Person目标通道。原P3图像资格与GT原序号保留，mask-valid差异显式记录。原来P5 `feature_drift_*` 日志不能替代完整P3类保持审计。

详细数学定义、实例资格、重叠处理与基准 pi_k=n_k/N 见 `S5_PARENT_FIDELITY.md`。实际Stage B记录包含GT/ROI有效性、各层特征维度、每类/尺度raw drift、Fourier梯度、保护冲突、同Frozen GT检测代理、真实5步与no-op对照。未测字段保留NOT_RUN，不从静态源码冒充实测层结构。
