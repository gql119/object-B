# 人工回复：验证集恢复授权

来源：用户对本会话验证集路径问题的直接回复，2026-10-25（香港时间）。以下为原文。

我目前只下载了 COCO minitrain 25K 训练集，Windows 本地路径为：
D:\迅雷下载\archive.zip
尚未下载 val2017.zip，也没有现成的 val2017 图像目录。
请继续执行以下任务：
1. 优先完成本地 archive.zip 向学校服务器的上传、解压、25K 图像验证和官方标注匹配，不要因为缺少 val2017 暂停训练集的数据准备工作。
2. 同时检查服务器是否已有可复用的完整 COCO val2017 5K 图像。
3. 如果不存在，优先尝试从 COCO 官方地址下载：https://images.cocodataset.org/zips/val2017.zip。
4. 如果服务器继续发生 502 或 SSL 错误，尝试 Hugging Face 公开镜像：https://huggingface.co/datasets/pcuenq/coco-2017-mirror/blob/main/val2017.zip。
5. 如果服务器仍无法下载，而你能够访问 Windows 本地环境，则下载到 D:\迅雷下载\val2017.zip，并通过已有 SSH/SFTP 连接上传服务器。
6. 如果你无法访问 Windows 本地 D 盘，请明确告诉我需要在本机执行的操作，并给出实际可执行的上传命令，不要编造服务器连接信息。
7. val2017 必须严格包含原始 5,000 张验证图像，并与已经校验的 instances_val2017.json 对应，不能使用训练集图片替代。
8. 数据全部验证完成后，恢复原始 S4 的 COCO Experiment A。继续使用原始 S4 的 8/4/0 扰动预算，不得使用 Full 版本，也不要覆盖已有实验结果。
优先并行处理训练集上传与验证集下载，避免不必要的等待。保留当前已完成工作，不重新下载 25K 训练集。
