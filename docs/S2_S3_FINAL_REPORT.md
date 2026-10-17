# S2/S3 Progressive Ablations — FINAL REPORT

唯一RunID s3_cnlp_cup_s0_eps8_person_seed0_20261015_v1。原流程exit0，rrctl completion health/wait/artifact contract/pull完成；科学源码017591686efb3675ee8d55debf1e2cd7da4e0e17、锁bd89244...、审查/准入/Parent身份匹配，运行期间未改科学设置，无重跑。

Change：S0+CNLP+CUP，原独立seed0初始化、Frozen/Adaptive/SafeBG/TSFL/Fourier/损失/预算8/4/0保持锁定；只在S2的CNLP聚合上加入预注册CUP。不是S2噪声继续训练。

Result：完整24epoch/9144Adamsteps，noise59217.920020624064s（16h26m58s），206checkpoint保留；Adam9144/scheduler24/CNLP9144/CUP9144和RNG/频率/query/模型状态验证。16551train/4952cleanval/13256PersonGT完整审计通过。随机初始化YOLOv8n200轮，连续results1..200且嵌入200行历史，train/val损失有限。最终checkpoint epoch=-1是optimizer strip语义，不冒充内部epoch200。lastSHA7b089c81f6d2244ed69990494cc211e20554e3f8edba013a9ad637a7dc1d03fb；bestSHA2c8ec15881511230dd2114dc953af8c3cae426eaabfb7b0603b6766bc4048b16与实际正式评价一致。

|last@200方法|PersonAP50|NTmAP50|NTmAP75|P|U|S|WCD|TCD3|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|S0|0.028993419333|0.691698089762|0.515898126283|0.966595258|0.902455927|0.933425073|0.281032380|0.257851168|
|S2|0.040368462714|0.689122622446|0.509452875723|0.953489512|0.899095725|0.925494090|0.308426301|0.274398588|
|S3|0.034413812012|0.689391325522|0.510290349028|0.960350157|0.899446301|0.928901002|0.269724805|0.260753864|

S3相比S0：PersonAP50 +0.542039个百分点（目标抑制较弱）；NTAP50 -0.230676pp、NTAP75 -0.560778pp，未建立整体效用改善。相比S2，PersonAP50 -0.595465pp、NTAP50 0.026870pp，后者幅度很小。S3最坏损伤类别为bicycle，WCD较S0改善但TCD3稍差，不能据此覆盖Parent规则或宣称全面优越。

S3次要best PersonAP50=0.039286534115、NTAP50=0.694457928054，仅作次要描述，不换主评价checkpoint。Person Recall=0.000441696113、Precision=0.490281738091是validator全局best-F1操作点；AP来自完整置信度曲线，不能混淆。完整AP50/AP75/AP50-95/R/P、target/NT/all、VOC20逐类值见S2_S3_COMPARISON.csv及S2_S3_VOC20_PER_CLASS.csv。

Finding：此单seed匹配协议中，CNLP/CUP未超过S0的目标与整体NT组合结果；S3比S2更接近S0，但无统计显著性或机制因果证明。不宣称背景吸收、消除Person-to-Dog、训练收敛，亦不从此前5step损失推出效用。原配置/失败证据/唯一独立审查均保留，没有新审查/调参/种子/额外baseline。

Next：S3已完整接受归档，执行后续已授权TFSS-C/U训练开发集有界准入与短对照；仍以S0为父，GPU资源重查，无其他用户竞争，完整TFSS噪声/victim需用户未来确认。所有数据/权重保留服务器，未删除或转移；原S1/S2/S3所有计算产物不可改。

证据入口：s3_archive/LOCAL_ACCEPTANCE.json及raw/，execution/s3_final_verification_20261017.json。P=1-AP50person/clean，U=NTmAP50/cleanNT，S=2PU/(P+U)；WCD=max signed relative NT drop，TCD3=top3平均，保留负损伤（改善）值。R/P与AP定义不混用。只归档小型JSON/CSV；未重跑评价。
