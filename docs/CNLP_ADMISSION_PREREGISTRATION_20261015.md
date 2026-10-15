# S0 Parent：CNLP/CUP 训练开发准入预注册

在任何新 GPU 观测之前记录。Parent 已按1pp规则锁S0；不使用S1扰动、CSB或poisoned victim初始化。S1原始结果只用于预注册父选择，不用于下面的参数标定。

阶段A（bounded calibration，不更新Fourier/不训练victim）：分别固定J=1/3/5，每个RunID执行38个train-only query对照case（19非目标类各2张，按`sha256('CNLP seed0 '+image_stem)`排序，排除实际support IDs），每case使用原Parent batch16/EOT2，CNLP support1/query1继承Adaptive batch1。Support取同一hash序下的6095张Person图像，按case循环选择16张；整个计划先落盘，三J一致。38outer-case/76EOT前向在每RunID100production-batch上限内；virtual SGD次数另报，不把虚拟更新当Fourier更新。

原Parent源码/45文件、config、reference checkpoint、原型及TSFL/SafeBG身份必须核验。保持Parent全部损失与初始化；lambda=1只测unit CNLP梯度，不执行outer optimizer step，不是正式权重。

Query所有非目标GT必须有共同native TAL positive/evidence；缺失即失败，记录该GT，不跳过/置零。不启动后续阶段，直到有明确训练开发证据支持的CNLP匹配修复。禁止更改Parent原损失。

J准入：所有case数值finite、全NT覆盖；同一共享匹配下clean query的median learning gain必须>0，且至少19/38 case的EOT平均CNLP梯度norm>1e-12。三J均测后选择满足条件的最小J，避免没有证据地增加短程复杂度。均不满足则停止并报告，不提高lambda或改变原扰动强度来制造梯度。

lambda预注册候选：只用选中J的38case平均EOT原Fourier变量梯度。非零CNLP样本集合上，`lambda = min(0.10*median(||gParent||)/median(||gCNLP||), 0.25*min(||gParent||/||gCNLP||))`，确保标定集每个非零case加权CNLP norm不超过Parent norm的25%，典型参考10%。零Parent/非零CNLP或非finite比值则阻断；不能用validation AP改规则。该比例是工程起点，不是理论最优。报告全部norm/方向/冲突频率及分位数；若attack余弦<-0.5超过一半有效case，先报告冲突风险，不正式运行。

CUP待训练开发审计候选：beta=.95、有效类别相对uniform权重界[.5,2]；tau取calibration正damage中位数，clip取绝对damage Q95，numerical floor=1e-8。参数在实际classwise/history/GPU sink审计通过之前仍非正式锁。全19非目标类必须出现在审计；缺席类不更新history，EOT仅在成功outer step后更新一次。不得按Dog/Horse/Bicycle的AP手调。若无正damage或rare-class/history/weight gate失败则阻断，不临时换公式。

阶段B：选中J/标定权重后，用真实Parent/new生产循环做lambda0原始Parent等价、S2/S3各5步及完整checkpoint/Adam/RNG/CUP/query continuation；检查像素8/4/0和label/数据隔离。随后对最终candidate进行一次科学实现审查，保留所有产物，不删除checkpoint。只有两配置同时锁定、独立review允许且资源通过后才正式S2→S3。此文件的候选值不得直接当作正式配置。
