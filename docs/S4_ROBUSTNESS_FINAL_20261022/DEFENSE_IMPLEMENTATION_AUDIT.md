# 防御实现依据

JPEG50/30共享同一函数，只变quality；Pillow固定subsampling2/optimizefalse/progressivefalse，uint8RGB一次encode-decode→PNG，无第二次JPEG。Gray使用OpenCV RGB→Gray→RGB整数实现。BDR从现有项目robust_transforms.py和run_robust_eval_modular.sh定位到5bit；float32归一化、31level、np.round ties-to-even及再次uint8量化均固定，作为检测训练数据适配，绝不声称此前S0已正式运行BDR。

Gauss/MeanF/MedianF严格来自实际S0 robustness_filter_audit.py；新PNG输出须逐像素匹配legacy函数。全部只变train，val指向原始clean且字节核验。防御后图像可能全局变化，8/4/0预算仅用于原S4身份核验，不能错误约束防御后图像与原clean的差。

来源：[Pillow JPEG参数](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg)，[Feature Squeezing作者页](https://www.cs.virginia.edu/~evans/pubs/ndss2018/)，[原论文](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-4_Xu_paper.pdf)，[REM官方实现](https://github.com/fshp971/robust-unlearnable-examples)。原论文是分类/推理防御依据，不能直接推出本轮检测训练效果。远端Pillow12.3.0/OpenCV5.0.0.93版本已记录；最新在线Pillow文档可能显示新版本，以实际锁定runtime为准。

AMP硬件自检复用历史防御local_amp_check：当前随机模型eval、固定zero1×3×256×256、FP32/autocast输出形状/有限性/allclose atol0.5，恢复train状态，不下载YOLO26n。该自检不执行优化更新；正式SGD/GradScaler继承原生。与S4离线跳过外部权重自检相比只增加显式自检证据，实际AMP必须为True，否则立即阻断；基线配对共享同一自检。
