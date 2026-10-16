# S0/S1 空间机制诊断

状态：**COMPLETE — MEASURED_CPU / CODE_VERIFIED**。推荐标签：`RECOMMEND_TSFL_ONLY`，仅作为下一阶段待批准实验候选；本轮至此停止。

## 核心判断

S1退化同时存在预算位置和优化算子的代理证据。它把仅由NT敏感度决定的预算施加到Person支持上，并用hard clipping替代原1/.5缩放；这既改变幅度分布，也改变有效梯度。不能由本诊断给最终AP做因果分解。正式 matched last@200：Person从2.89934%升至9.16705%，是 **+6.2677个百分点**；NT mAP50为+1.2878个百分点。保护NT变好不保证Person UE维持原强度。

## A：TSFL与保护位置 — H1 MIXED

65个对象归因均有限非零；每对象未裁剪归因落在自身框内的质量均值 **31.12%**，32图合并后落在Person union的质量均值 **43.95%**。这是两个不同口径。选定通道的感受野明显延伸到框外，不能把通道特异性直接等同于像素类别隔离。

图内TSFL Q与原完整保护梯度的Spearman均值 **0.3436**，Top20集合重合率 **36.00%**（30张有效图；独立均匀选择的参考约20%，不是显著性检验）。与S0历史实际PNG幅度的相关仅 **-0.0530**；Top10/20 Q区域拥有S0能量 **9.24%/18.85%**。归因与梯度有中等对应，但不是“原S0高幅度区域”的重命名，也未证明可靠的UE收益图。

几何面积校正后：中心/边缘TSFL质量密度相对均匀分布为 **1.0596/0.9719**；保护梯度能量密度为 **1.0226/0.9999**；S0实际扰动能量密度为 **0.9664/1.0179**。没有明显的“框中心普遍更重要”证据。仅说边缘拥有约49%归因会误导，因为边缘定义r>=.8本来占约一半框面积。

对象尺度的TSFL/保护相关：small0.2998（5GT）、medium0.3459（32有效GT）、large0.3380（26GT）；逐图Person-only0.2281（8）、Dog0.4524（3）、Horse0.3911（5）、Bicycle0.5058（4）、多人0.3330（15）。类别/尺度分层重叠且样本少，不能据此设类别权重。完整数据见 `OBJECT_SCALE_SUMMARY.csv`、`STRATUM_SUMMARY.csv`。

两张图005439、2011_001227的原保护梯度空间图为零，位置重合/保留率不作零值证据；30张进入相关聚合。2009_000455的两个GT feature-mask mass仅0.58594/0.32813，低于原池化门槛；归因仍有效，但不能冒充原完整对象监督。65GT归因、63GT有效保护空间统计、30图相关、28图BG代理和24图NT代理的分母分别保留。

## B：裁剪确实改变梯度 — H2 SUPPORTED（限定配对条件）

同一个恢复的S0 final-active Fourier状态、同batch/损失/EOT下，使用真实S1投影；其像素Jacobian与生产autograd每图每EOT最大差0。无NT的8张图走原始legacy乘法分支，不能当作hard clip样本。

|算子|阻断的g_out标量通道能量|相对a0*g_out的范数保留|方向余弦|
|---|---:|---:|---:|
|S1 literal CSB|57.67%|68.69%|0.5997|
|S0同预算纯hard clip对照|23.27%|69.55%|0.8472|
|同CSB cap乘法对照|0.00%|94.07%|0.9695|

能量分母是原g_out，范数分母是a0*g_out，不能互相相减或由57.7%直接推出范数只余42.3%。逐EOT再图均值，30张非零梯度图；局部像素比例用全32张。严格越界RGB标量比例58.21%，全部RGB同时Jacobian为零的像素比例36.80%。高TSFL/高保护位置阻断能量分别58.41%/58.00%。只计算全部RGB归零像素的能量会低估为36.96%，故与标量通道量分列。

同基础输入的operator-only Fourier VJP，clip/scale范数比 **0.9834**，余弦 **0.8363**（2batch/EOT平均）。范数接近不表示像素梯度未被阻断；共享Fourier方向/相消、重叠区倍率及raw TV/budget旁路均可掩盖局部截断。实际完整Fourier梯度另见 `PAIR_BATCH_LOSSES.csv`。

边界min/max导数是0.5，严格越界0，内部1；原上游全局clamp和图像clamp仍在。2.01/255 floor与约2/255最小CSB cap相遇会保证局部阻断，但本样本仅覆盖框内像素图均值 **0.92%**，对应平均EOT保护梯度能量 **1.64%**。它只是小部分贡献，不能被包装为主要原因。

## C：预算与目标保护敏感位置有冲突

原始G_NT与Q的相关0.3381（24图），与保护梯度相关0.2022（22图）。高保护Top20位置中 **54.79%** 被分到低于S0的预算；均值下降0.5270 byte，非Top20位置平均下降0.1218 byte。高TSFL位置约55.16%预算下降。重叠与独占区域的高保护位置预算下降比例分别45.06%/71.52%，不应合并解释成均匀减弱。

固定样本CSB image-mean高NT(norm>=.9)/低NT(norm<=.1)预算2.2582/7.8020 byte，重叠5.1514、独占7.0423，与历史全6095图的2.2504/7.8159、4.9997/7.0592口径一致但样本不同。当前整体CSB image-mean6.6125，pixel-weighted7.1762；S0对应6.7913/7.3006。分母、像素数和完整配对见 `BUDGET_DOMAIN_WEIGHTING.csv`，不能跨权重混比。

历史PNG的S0/S1框内RMS逐图均值0.026253/0.024282，是不同已优化参数的观察结果；不能用它独立推断算子的因果作用。本轮paired gradient的S1算子输入是同一S0基础扰动，不是S1历史每一步状态。

## D：三种软缩放 — H3 MIXED，B适合作为最简后续候选

参数在运行前固定：eta_TFSS=.5、sigma=.5、rho=.5，仅各一个候选；保留原a0、全部原损失与上游合成顺序。以下RMS/MAD与能量均使用**图像clamp之后**实际poison-input；逐图比例再等权平均。强度32图，保护位置/梯度30有效图。

|方案|框内均值权重|实际RMS/S0|实际MAD/S0|高保护区实际能量/S0|固定g_out局部梯度范数/S0|
|---|---:|---:|---:|---:|---:|
|Center-only|0.6805|68.93%|68.03%|48.83%|69.56%|
|TSFL-only|0.6854|69.84%|68.35%|60.42%|79.00%|
|Fusion|0.7628|76.48%|76.21%|65.04%|81.23%|
|UniformB|0.6974|69.89%|69.96%|48.69%|69.64%|

B与UniformB请求能量配对最大相对差2.904e-7；实际clamp后能量最大偏差1.869%。两者平均实际RMS几乎一致（0.016873 vs0.016887），而B在高保护位置保留60.42%能量、Uniform48.69%；固定g_out梯度保留79.00% vs69.64%。这是空间分配有用的静态代理证据，但不是完整强度控制或UE效果证据。

C的实际RMS保留76.48%，明显大于B69.84%；其有效最低权重0.625、B为0.5。C的较高梯度保留不能直接归因于中心互补性。B的框内相邻像素权重TV均值0.03518，大于A0.00645与C0.01800；“soft”表示额外乘法导数非零，不等于最平滑，也没有移除原所有梯度饱和。

不隐瞒负面代理结果：相对S0，target-relation代理A/B/C/Uniform均值变化+0.01928/+0.01392/+0.01080/+0.01657，**均更差**（越小越好）。BG代理均值B为-0.00180，但中位数+0.00205，变化不一致。NT feature drift相对S0的减少量B0.03891、Uniform0.07005，故B对NT漂移的改善小于均匀减弱。NT检测代理B均值+0.000699，中位数-0.00000677，不能声称普遍改善，更不能当AP。全部逐图/EOT差异在 `PAIRED_PROXY_DIFFERENCES.csv`。

参数级完整Fourier梯度余弦A/B/C/Uniform为0.6682/0.7797/0.8710/0.7275（2batch），说明方向仍改变。没有测最终频谱分布、长期优化、收敛、UE效力、因果背景吸收或Person-to-Dog变化；这些是UNVERIFIED。原始Fourier参数化保留，不等于最终空间调制后频谱完全保留。

## 范围与执行结论

MEASURED：32train/65GT原始归因、真原S0元梯度、配对算子及全部小表，6个按GT分层固定选择的代表图及独立对象图。CODE_VERIFIED：源/权重/冻结图身份、真实分支及Jacobian、预算/模型不可变门槛。HYPOTHESIS：上述冲突可能贡献S1 Person退化。UNVERIFIED：贡献比例、最终AP/多seed、TFSS正式效果和正式GPU开销。

可选5步优化为NOT_RUN；没有必要为静态机制证据虚构历史Adam/RNG或引入额外状态。没有训练、生成数据集或victim，没有数据/权重转移、删除或GitHub推送。S0/S1/S2/S3未改动。所有图均为派生热图，未传原始RGB数据；青色Person、品红NT，原始大小坐标，P99仅影响显示色阶，不影响CSV统计。初版标题重叠已修正，旧图保留在 `figure_qa_v1/`。

本轮不启动TFSS。15问与最终候选/风险见 `FINAL_RECOMMENDATION.md`。

## 可追溯身份与口径

- 诊断源：`5bcae7233e03ebc6d2533f0ae311f67a68676899`；RunID `tsfl_spatial_cpu_fixed32_seed0_20261016_v1`。完整原始证据在 `raw/`；聚合见 `AGGREGATE.csv` 和 `AGGREGATION_POLICY.json`。
- Frozen/Adaptive Reference SHA：`8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89`；S0 final Fourier SHA：`2490dd1ef1b95d1a1df9b8f32ce69555ec2516f55eac832ef6ff40003c1f2a77`。S0 当时 runtime Git SHA 未知；不把后来源码归档当作当时提交。
- 数据：VOC_0712_Kaggle_Ready; original clean train manifest f38fa031946d9eaec26665b09fcfebad09c2c1e9e20deff3b5770192eb5cef8f；固定训练32张/65PersonGT，seed0。完整 Image ID/GT/图像与标签SHA见 `FIXED_TRAIN_SAMPLE_MANIFEST.json`，committed-LF SHA `d15881f71f8c62ad0624f1a9c8153c0a146511978631df2017ec323373ad3522`。没有使用 validation 选样、选参数或优化。
- 特征：原始 P5/model.21，24通道 `[0, 6, 10, 29, 41, 68, 77, 86, 89, 112, 116, 125, 128, 133, 140, 167, 173, 184, 220, 229, 237, 238, 239, 240]`；原通道选择工件未重建。归因目标为每个 Person GT 的选定通道**平方响应均值**，使用原 fractional feature mask；RGB输入梯度L2。先3×3平滑，再各GT框内Q10/Q90归一化，多Person取max。
- 原图坐标；batch内右/下pad至32倍数，沿原实现letterbox至640，再做原feature mask映射；统计图裁回原尺寸。细节与源码身份见 `SOURCE_AUDIT.md` / `SOURCE_METADATA.json` / `PROVENANCE.json`。
- 实际设备 CPU FP32、单线程、nice19、CUDA隐藏；耗时 `570.122s`。batch16/EOT2，模型及Fourier参数未变，optimizer updates0。真实原S0 Frozen损失和Adaptive高阶路径均执行；不是普通分类梯度。`g_protect=||mean_EOT(g_out)||_RGB`；局部Jacobian能量逐EOT计算。
- EOT先在同一图内平均，再逐图等权；对象表按GT计数；共享Fourier余弦按2个batch计数。预算另给pixel-weighted结果；二者不能混用。32张为分层开发样本，不代表总体分布，不作显著性或因果检验。
- 单图/修复probe及失败统计保留。唯一独立审查原结论为 scientifically_incorrect；B1–B3及主代理发现的B4已由当前源真实CPU probe闭环，**没有第二次审查或虚构独立通过**。见 `execution/lean_review/MAIN_AGENT_CLOSURE.json`。
