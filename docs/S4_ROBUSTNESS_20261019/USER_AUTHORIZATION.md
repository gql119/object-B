# S4 鲁棒性正式授权记录

来源：2026-10-19 本聊天中用户直接提交的《Codex 正式研究任务：S4-TFSS-Utility 目标检测不可学习样本鲁棒性实验》20节；本文是执行范围摘要，不声称逐字副本。

明确授权 Phase A JPEG50→JPEG30→Gray→BDR（有源码/文献依据才运行），随后 Phase B Gauss→MeanF→MedianF→Data Augmentation 的完整训练评价与独立 GitHub 分支发布。固定 S4 既有原始加噪数据；不重新优化扰动、不使用任何 S0/S3/S5 poison。每项配对 Clean/Poison；四项历史滤波/增强 Clean 优先直接复用已核验200epoch last，不额外重训。新 Clean 只在无可复用对照时执行。训练200/seed0/batch36/workers16/AMPtrue/nativeSGD/last@200primary，验证始终未变换的4952cleanVOC。禁止修改冻结S4科学定义/参数，禁止AP调参/额外种子/并行正式GPU训练。允许必要数据和实际训练smoke。AT只审计协议与成本，未经后续人类确认不得正式训练。保留全部失败、checkpoint和历史数据，不自动删除，不干扰其他用户。

用户指定：若 BDR 定义/参数无法确定，标记 BDR_PROTOCOL_UNRESOLVED，不猜参数；其他已确定项目继续。当前从保留的项目源码和shell明确找到5-bit实现，另有Feature Squeezing原论文5-bit设置依据；将其标记为检测训练数据预处理适配，不声称原文的检测协议。
