# 2026-10-15 当前CUP提案（以下旧草案作为历史）

beta=.95，temperature=0.0003758072853088379，damage_clip=0.0010146975517272938，有效类相对uniform权重[.5,2]。来自固定train-only损伤的预注册median/Q95规则，未用validation AP挑类或调参。与S2同Parent/SameCNLP/J1/lambda0.058485507823134296，S3独立原Fourier初始化，不接续S2噪声。历史全零起始，第一次观察直接初始化，缺席不变；每个成功Adam step之后提交一次EOT平均损伤，weights使用pre-step detached EMA，bounded simplex严格sum1。

实际GPU作用：10步/20EOT固定多类别query中12批的weighted-vs-uniform Fourier梯度不同，CUP steps10；只证明sink可达，不证明最终utility/AP、稀有类性能或Person-to-Dog FP改善。真实5步/3→5恢复checkpoint差0/日志byte-identical；完整dataset/200epoch victim仍NOT_RUN。19类真实train damage CPU replay通过，属于CPU replay，不是19类GPU训练。

---

# CUP 实现草案

状态：DRAFT；正式 CUP 参数和真实训练类间尺度/少样本审计 NOT_RUN。

复用 S2 完全相同的 CNLP 双路径、query、d_k、J、lambda 和 Parent。S3 独立从原始 Parent 扰动初始化开始，不继承 S2 最终噪声。CUP 只把均匀类别聚合替换为 damage-history 加权。

首版候选采用上一完整 optimizer step 的 detached signed damage EMA。每次 Adam 成功后，对该步各 EOT 的同类 damage 先求均值、按明确配置 clip，再更新一次。只更新出现的类别，缺席类别不衰减；首次观测直接初始化，避免零初始化偏差。history、观测数与完成 step 一同检查点化。step 尚未完成时拒绝导出可恢复状态。

当前有效类别上计算 `softmax(m/tau)`，用 bounded simplex 的欧氏投影保证和为 1，且每类权重在 `[min_factor/N_valid,max_factor/N_valid]`。直接 clamp 后归一化会破坏上下界，因此没有采用该实现。min_factor>0 提供基本保护；max_factor 限制单类集中。没有手动 Dog/Horse/Bicycle 权重，也不读取正式验证 AP。S3 的历史权重与新增损失均保留相同 ReLU(d_k)。

beta、tau、clip、min/max 因子都由显式构造参数传入，配置草案中为 null。CPU 测试夹具 .8/.2/.5/2/5 只验证代数与恢复，不是推荐值、预注册值或正式配置。少样本第一观测仍可能影响权重，当前仅有 clip/权重上限的数值约束；必须进一步做真实训练开发稳定性审计，不能宣称已解决少样本偏差。

真实 NT GT 损失保持不能保证去除 Person→Dog FP。本实现没有加入 E1-XFP、E2-Localization 或 KD。
