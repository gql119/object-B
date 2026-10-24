# COCO minitrain 与原始 val2017 图像审计

状态：DATA_READY。训练25,000/25,000，验证5,000/5,000均逐张完整JPEG解码、尺寸核对、CRC、写出SHA读回以及官方JSON派生标签审计通过。固定官方ID缺失0、多余0，train/val无交集，未重新采样或重编码图像。

| 项目 | train25K | clean val2017 5K |
|---|---:|---:|
| 图像/标签文件 | 25,000 / 25,000 | 5,000 / 5,000 |
| 原始JSON实例 | 183546 | 36781 |
| YOLO非crowd/ignore实例 | 181475 | 36335 |
| Person GT | 55096 | 10777 |
| 空标签图像 | 0 | 48 |

训练含Person图像13630张、无Person图像11370张，均完整保留。COCO80官方稀疏category_id按升序映射到连续YOLO ID，Person=0，80类及逐类实例数核对通过；不使用Kaggle标签重建映射（该包无检测标签）。bbox有效，按原始宽高归一化，不裁剪/删除实例作为修复；crowd/ignore明确不写入YOLO标签，官方JSON原样保留用于正式COCOeval。

训练官方JSON SHA256：`5682389c3bb0a356912771811f7836659cc6de5a588413752922ccb94a6df1a8`；验证官方JSON SHA256：`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。服务端根：`/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023`，图像分别在`images/train`与`images/val`，标签在`labels/train`与`labels/val`。

全量每图/标签SHA清单：训练 `/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023/metadata/TRAIN25K_KAGGLE_IMAGE_MANIFEST_20261024.jsonl`（SHA `a8bd9bd336685bba1573d0401ad686f76e156daf35ea9867499a34c2fb31a343`）；验证 `/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023/metadata/VAL2017_IMAGE_MANIFEST_20261025.jsonl`（SHA `6fef2d07ef94fbc190519f44959eacdca0139d551514c671bf42843d33b18f67`）。本地已读回并独立核对完整ID集合及实例总数；原始图像只在服务端全量审计，未再次拉取图像至Windows。

数据READY不代表COCO算法生产入口、GPU smoke、科学审查或正式训练已完成；本次没有GPU初始化或正式科研指标。

主agent最终独立证明：全部30,000张图像的本地原ZIP成员SHA与服务器实际输出清单逐项相同；全部30,000个标签SHA与官方JSON的既有CPU契约转换逐项相同。训练与验证ID严格无交集，80类映射及Person0正确，原方法源码和冻结优化参数未因数据恢复改变。未重复服务器全图扫描、未拉回图像或生成GPU/正式指标。完整证据见COCO_DATA_FINAL_ACCEPTANCE.json。

报告最终核验时间（香港）：2026-10-25T01:21:54.319857+08:00。
