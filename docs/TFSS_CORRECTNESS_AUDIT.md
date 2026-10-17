# 当前正确性审计

|项目|实际状态|
|---|---|
|隔离源码归档/远端重新读取|PASS，a5aefbb1376cd5aacc0e2cefd73cebdbcf808f54|
|S0历史45文件SHA|PASS，运行时GitSHA未知|
|C/U公式、无NT退化、原1/.5/0预算|CPU组件PASS|
|裁剪后RMS配对、超出可达域失败|CPU组件PASS|
|软缩放、地图detach、Adaptive实际poison缓存|CPU组件PASS|
|诊断梯度保留图、完整backward仍可执行|CPU高阶数值探针PASS；非VOC模型|
|真实VOC S0 loss/Fourier/single-step等价|NOT_RUN|
|真实nativeTAL/所有NTGT覆盖|NOT_RUN|
|真实CUDA高阶图/模型状态/资源测试|NOT_RUN|
|新TFSS独立科学审查|NOT_RUN，等待当前commit GPU smoke|
|GPU5-step/逐类/热图/曲线|NOT_RUN|
|完整噪声/数据集/victim/正式AP|NOT_RUN且未授权|

本地Torch2.14.0+cpu用于轻量数值检查，服务器已知Torch2.7.1+cu128/Ultralytics8.4.150；不安装/更改环境，不声称本地数值代表server bitwise。首个新增序列化测试的C++ filename writer失败证据保留为CPU_VALIDATION_INITIAL_FILE_IO_FAILURE.json；改用Python二进制文件句柄并保留owned tiny artifacts后测试通过。未删除文件。全部失败与未运行项如实保留。

S3源目录/RunSpec/配置/receipt/checkpoint从未修改；只有只读rrctl观察。唯一物理GPU已被S3占用，所以没有上传TFSS脚本、没有新远程进程或模型前后向。源码独立是可验证事实；不把一段观察夸大成整个共享服务器零影响的因果证明。


## 2026-10-17 latest measured status — overrides earlier NOT_RUN / WAIT_FOR_S3

S3 fully accepted/archived/exited before TFSS started. Source897cdfe smokev2 passed actual canonical S0/noop equivalence maxdiff0; S0-reference/noop/C/U each completed1update/EOT2 on firstfixed16 with unchanged model hashes. It then failed the locked Uniform postclamp RMS error1e-9 gate; first complete six-method smoke and both-batch5step remain INCOMPLETE. No reviewer called and no formal TFSS. CPU saved-delta replay finds nearest float32 bracket error>1e-9 for11/16 first-EOT images, maximum7.5655e-9. Full evidence and a concrete **unapproved** numerical protocol proposal: [RMS_GATE_BLOCKER_20261017.md](tfss/RMS_GATE_BLOCKER_20261017.md). WAIT_FOR_HUMAN_NUMERICAL_PROTOCOL; no automatic retry or tolerance change. Original source45/formulas/budgets/eta/losses/optimizer/sample/EOT unchanged.


2026-10-17 NUMERICAL_STABILITY/PROTOCOL_AMENDMENT（人类明确批准）：Uniform选择最接近的可达FP32 alpha端点，保持[0,1]与原FP32前向，裁剪后实际RMS误差同时满足absolute<=1e-7及relative<=1e-4，配置值必须到达实际求解器；实际误差逐图保存，任一门失败即停止。原1e-9检查及未批准/等待文字仅为保留的历史；配置/样本/eta/预算/损失/优化器及S0/noop1e-7不变。新测试覆盖浮点台阶、绝对与相对门独立失败。源码需新RunID smoke通过后再一次独立审查。
