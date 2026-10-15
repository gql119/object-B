# 2026-10-15 状态覆盖

S1已完整验收并归档（docs/s1_archive/），Parent按原1pp规则锁S0（parent_selection.json）。J1v2/J3v2固定38case/76EOT标定数值/学习门槛通过，J5v2已唯一启动，其观察连接出现SSH超时，尚未读取完整结果，不能记为失败或完成。三J全部取得后才应用预注册规则。新增生产准入/正式入口仍未部署/未GPU运行；两正式配置、最终科学审查、S2/S3完整实验均NOT_RUN。候选commit ca57a71仅绑定已启动J5，工作树后续新文件不是当前远程运行代码。

原下文是2026-10-14准备历史，UNSET/NOT_RUN字段由上文当前覆盖；不可误认为S1仍在跑。研发只在隔离S23树，原Parent/S1运行文件和结果不变。当前没有GitHub发布新CNLP代码。

---

# S2/S3 代码准备（2026-10-14）

状态：LOCAL_IMPLEMENTATION_TESTED；Parent=UNSET；真实 VOC/GPU 准入=NOT_RUN；S2/S3 正式实验=NOT_RUN。

用户在原分阶段任务之后明确要求“先写相关代码”，因此只提前实施独立本地代码和 CPU 正确性测试。原 Stage 0 的 S1 运行保护、正式父版本选择和实验准入继续有效。没有更改或停止 S1，没有学校服务器写入、新 GPU 任务、Git commit 或 GitHub push。

工作树：`safe_soft05_s2_s3_counterfactual_20261014`；分支：`codex/safe-soft05-s2-s3-counterfactual-20261014`。工程起点为 `16e0733227e32910cf8b480bd3497dff169c92f3`，不是选择 S1 为 Parent。原 S1 未提交修改未复制、未覆盖。

## 实现

- `counterfactual_utility.py`：两个相同初值的 functional SGD 虚拟更新，poison 保留高阶图，clean 路径 detached；BN running statistics 冻结，双路径重放随机状态，并恢复调用前全局 RNG。
- `counterfactual_yolo.py`：原生 YOLOv8 all-GT support loss；同一 clean train query，使用 clean-updated 路径的 detached native TAL assignment，在两个路径共享 GT/anchor/soft target；逐对象 box/BCE19/DFL，再逐类别均值。合法无 NT GT 返回连通零损失，存在 NT GT 但缺匹配则报错。
- `counterfactual_extension.py`：Parent + lambda_CNLP * CNLP；CUP 只替换类别聚合。历史在成功 outer Adam step 后提交一次，EOT 内不更新历史。
- `counterfactual_trainer.py`：独立 S0/S1 trainer 包装器，S0 直接继承原 B2Semantic，S1 继承原 S1CSB；测量原 Fourier 变量上的 CNLP 原始/加权梯度及与 Parent、attack、ret 的余弦。零范数余弦配有 validity mask。
- `counterfactual_query.py`：manifest SHA/图像及标签指纹校验、train/val 身份隔离、排除整个 Parent support batch、可恢复的确定性 train query 计划。
- `counterfactual_optimization.py`：独立优化循环，保留 Parent Adam/EOT/frequency/scheduler 更新规则，新增 CNLP/CUP/query 状态保存及恢复。正式入口要求 LOCKED 配置；没有实际正式启动器。
- `counterfactual_parent.py`：完整验收及匹配 last.pt@200 协议检查、精确 1 pp 双条件选择、S2/S3 单变量配置比较。现在没有生成真实 parent_selection.json。
- `prepare_counterfactual_inputs.py`：只读 train manifest 审计和已完成父版本证据绑定，不训练；输出独占创建，不覆盖已有证据。

## 已完成验证

55 项本地 CPU 测试通过：47 项新测试、8 项原 Parent/CSB 契约回归。合成 YOLOv8n/VOC20 head 验证 J=1/3/5 的 native forward、共享 TAL、非零二阶扰动梯度和零扰动双路径相等。标量回归有限差分绝对误差最大 `1.965375084900245e-11`。合成外层连续/恢复比较的 Fourier 参数、CUP 历史、Adam、scheduler、Torch RNG 和 diagnostics CSV 完全一致；完成状态恢复没有额外更新。

一项警告来自未改动的历史 `test_b2_adaptive.py` 将 requires_grad tensor 转成 float；不是新代码运行失败。测试临时产物保留在忽略的 `.local_runtime/`；没有清理。所有测试输入与超参数只是测试夹具，不是 VOC 科研结果、正式标定值或科学审查通过。

## 尚未完成 / 硬准入

1. S1 完整接受和归档；匹配 S0/S1 的 last.pt@200 评价，按两项 1 pp 规则选择 Parent，绑定真实 source/config/checkpoint/budget。
2. 实际 VOC train manifest、support/query 和 NT GT 匹配覆盖审计。当前严格 native TAL 覆盖可能失败，不能用零损失掩盖，也没有自动 rematch。
3. J、支持/query batch、检测分量/类别归一化、lambda_CNLP 的训练开发数据标定；CUP beta/tau/上下界/damage clip 和少样本行为审计。全部 UNSET。
4. 实际 Parent + 新 trainer/完整循环的生产等价、梯度方向/范数分位数、GPU 显存/耗时和真实断点恢复。CPU mock loop 不能替代这些准入。
5. 依据适用技能完成真实预运行检查和审查；不得把当前 CPU 通过当正式科学准入。
6. 锁定两份配置和完整训练协议，顺序执行独立 S2 → S3，完整审计/随机初始化 victim200/last 主评价/best 次评价/全部 VOC20 指标及最终报告。

没有做 validation AP 调参，没有选择 Parent，没有修改原 L_ret、Safe-BG、L_orth、CSB 或原扰动参数化。测试均已落盘，但不能据此宣称 CNLP/CUP 保持了真实非目标学习能力。
