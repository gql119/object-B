# COCO 本地 ZIP 传输

状态：训练 25K 与验证 5K 均完成传输、双端完整 SHA256 比对及图像审计。更新：2026-10-24T17:21:38.196803+00:00。

| 项目 | 用户本地 Kaggle 训练 ZIP | 独立官方 val2017 ZIP |
|---|---|---|
| Windows 路径 | `D:/迅雷下载/archive.zip` | `D:\迅雷下载\val2017.zip` |
| 字节数 | 4,066,483,999 | 815,585,330 |
| SHA256 | `1944317fa24e3f1c75bbd28159bf8246c8a93fc5a0f3fefbd4bf32e005404040` | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| 本地/服务端完整 SHA256 | 完全一致 | 完全一致 |
| 成员 CRC/目录安全检查 | 全部通过 | 全部通过 |
| 固定官方 Image ID | 25,000，缺失0/多余0 | 5,000，缺失0/多余0 |

服务端训练压缩包：`/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023/archives/kaggle_coco25k_1944317fa24e3f1c.zip`。验证包：`/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023/archives/official_val2017_4f7e2ccb2866ec50.zip`。压缩包全部保留，先独立 `.part` 接收，逐32 MiB哈希验证，整包 SHA 一致后才重命名。现有 RRCTL SSH 通道已验证；Windows 原生 ssh/scp/sftp 可用，PATH 未发现 rsync/WinSCP，实际使用现有 RRCTL 的有界内存/前缀校验续传实现，未宣称运行过独立 SFTP 测试。

训练 ZIP 的全25,001成员CRC、25K ID及路径检查在本地已通过；其中两张 JPEG 另与留存的官方来源哈希一致（样本证据，不外推全量像素来源证明）。[作者仓库](https://github.com/giddyyupp/coco-minitrain)列出了 [Kaggle coco25k 镜像](https://www.kaggle.com/datasets/trungit/coco25k)。ZIP哈希证明本次传输身份，不要求与其他镜像压缩包哈希相同。

验证集从用户指定的 Hugging Face 公开镜像 `https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/main/val2017.zip?download=true` 单独获取，严格TLS、精确Content-Range与公开完整SHA256校验，原生curl最多4连接、512KiB分段、每请求55秒/至多2次尝试，保留失败及续传修复记录。不重新下载训练集、不在C盘解压、不覆盖官方标注/VOC/旧Poisoned数据、不安装环境或删除产物。原代码Snapshot已先完成：`f00ef6062557bef0f3b5081702de1e52ba9beb0a`。

报告最终核验时间（香港）：2026-10-25T01:21:54.319857+08:00。
