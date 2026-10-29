# COCO Experiment A 最终结果

Original S4 — COCO80 Dimension-Compatible Implementation + Level C NT-risk Boundary Extension。
协议：`coco80-ntgt-native-tal-then-gt-risk-v1`；`strict_Original_S4_equivalence_claim=false`。

原唯一控制器已完成全部9阶段，RRCTL终态为completed、exit0。所有主结果来自独立随机初始化seed0的YOLOv8n，固定last.pt@200，在完整干净val2017 5000张/80类上评价。

| 主结果 | Person AP50 | NT79 mAP50 |
|---|---:|---:|
| Clean200 | 71.197106% | 39.114393% |
| S4/Poison200 | 10.360167% | 30.858504% |

P=0.8544861241891832；U=0.7889296464142012；S=0.8204003489328877。
Person绝对AP50差值=60.836939个百分点。
WCD/TCD3为null：Clean AP50为0的非目标类为['hair drier']，完整79类比值不可定义。
LPIPS未验证、未测量，保持null。PSNR沿用已验收导出诊断。
这是固定单seed的描述性比较，不作因果或统计显著性结论。

正式评估/最终报告源码：349720570a8f388acb7358cc689f375728dd6c3e。
正式评估receipt SHA：4a147e702ae7d44848ec7ed2c35d028f84adc75a8fabdfb78250fc063b09f6c3。
最终报告receipt SHA：016f4889cc1375ee970259dbdb4a68560ce367dfc1a13e346931b0a0f0ca6d89。
最终报告SHA：119ab329e18b11d33d5aad3ded3d8babeb94f0b96bf87d376a675d2dd74b63bb。

原official-eval独立负面评审保持原文。主agent关闭B1配对指标缺项、B2验证图字节身份、G1协议归属；新CPU验证和GPU03实际CUDA复验/5000图身份回执通过。GPU02清单字段接口失败保留，未复用RunID。GPU03两checkpoint各36图、8次总原生前向、原生输出差异0/容差0。

本地Python3.12对U/S的sum归约比服务器3.11相差2.22e-16；没有新增容差，实际发布源码/服务器运行时精确复算通过。诊断与主审计均保留。

本目录只归档小报告与验收证据，不含图像、权重、预测大文件、profiles或秘密。
用户2026-10-29最新授权：COCO完成后继续VOC Original S4 AT4/AT8四组串行实验，仍须实际CUDA、一次科学审查、主closure和资源门。
