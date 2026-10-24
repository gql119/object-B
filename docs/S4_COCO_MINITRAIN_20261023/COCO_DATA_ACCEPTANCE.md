# COCO 数据接受状态

训练25,000张和原始clean val2017 5,000张全部通过实际服务器JPEG完整解码、CRC、尺寸、官方ID和官方JSON标签核对。主agent进一步独立核对全部30,000个本地原ZIP JPEG SHA及30,000个官方JSON规范转换标签SHA，与服务器读回清单逐项相同；train/val ID无交集，80类/Person0正确。训练181475有效GT/55096Person GT，验证36335有效GT/10777Person GT/48空标签均保留。

证据：[最终独立证明](COCO_DATA_FINAL_ACCEPTANCE.json)、[图像与标签审计](COCO_MINITRAIN_IMAGE_AUDIT.md)、[数据清单](COCO_MINITRAIN_MANIFEST.json)、[类别映射](COCO_CATEGORY_MAPPING.json)、[传输证明](COCO_LOCAL_ZIP_TRANSFER_REPORT.md)。本数据阶段已完成，不重复下载、上传、解压或全图审计。已接受的数据不代表完整算法/生产入口/GPU/科学审查已通过。
