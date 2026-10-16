# INSUFFICIENT_EVIDENCE

本轮尚无真实5-step对照，不能选择TFSS-C或TFSS-U进入正式200epoch实验。源码/CPU正确性仅为部分Gate A，不能证明目标保持或非目标收益。eta_C/U均固定.85，无参数搜索。

最终16问当前答复：

1. 当前实验代码已隔离提交并推送object-B新分支，SHA a5aefbb1376cd5aacc0e2cefd73cebdbcf808f54。
2. git ls-remote重新读取同SHA；main仍b89147ee0e46cea3fbba10a8b6bd9af7e60a5b43。
3. S3目录/运行参数/检查点未修改，无新共享GPU任务；完整“零影响”不作因果保证。
4. 从按实际历史45个文件SHA恢复的S0开发；不是从S1/S2/S3方法开发。
5. 原loss/优化器计算体未改；运行等价仍NOT_RUN。
6. C保护梯度保持：NOT_RUN。
7. U减少NT干扰是否胜C：NOT_RUN。
8. 两者实际RMS下降：NOT_RUN，不能用理论0.85floor代替测量。
9. 优于各自Uniform：NOT_RUN。
10. U风险额外开销：未测；已分别记录预备计时。
11. 谁更接近S0目标保护：未知。
12. 谁NT效用更好：未知。
13. 跨样本稳定性：未测。
14. 类别间得失：未测。
15. 当前不推荐任何正式200epoch候选。
16. 最低下一步：S3完成后执行已锁定训练32/两个batch/各5step及对应RMS对照，先完成真实父版本等价与当前源码smoke/单次审查；不必先跑完整victim。

Change/Result/Finding/Next见SHORT_PROBE_RESULTS；所有可疑、undefined或冲突证据保留，可最终继续KEEP_S0/INSUFFICIENT_EVIDENCE，禁止靠反复调参制造成功。
