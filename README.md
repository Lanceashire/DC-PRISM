# DC-PRISM

独立的新实验仓库：研究延迟选择性反馈下，风险保护、Pending-aware 信息选择和标签数量控制的作用。以本次提供的新源码为主线；[LexiRiskLabel](https://github.com/Lanceashire/LexiRiskLabel/tree/b4f3f2db6788bb8c7bb127a209e4f518fbafe226) 仅作方法与历史证据参考。

本仓库同时保留提交包原始证据与经过修复的实现。完整版本关系、已修复项、仍有限制的主张见 [严肃代码审查](docs/CODE_AUDIT.zh.md)，两段聊天与整合报告的处理见 [整合决策](docs/INTEGRATION.zh.md)。

- `src/`：修复后的 P3/P4 实现与安全运行入口。
- `tests/`：原有完整性检查及新增的求解、隔离、动态释放回归测试。
- `archive/submitted_v2_2/`：按属性、内容和结果覆盖确定的最新提交包原始源码、合成证据和聚合表，保持原样。
- `results/repair_v1/`：修复后按原 16 seeds 重跑的聚合结果。属于协议修复后的再运行，不是新的独立确认。
- `provenance/`：输入 SHA-256、本地文件属性、ZIP 内时间、版本差异、原结果重建证据与运行清单。
- `reference/`：旧仓库固定提交的参考文件；不向新项目施加旧研究的阶段规则。

原提交 P4 已在 NumPy 2.1.3 / SciPy 1.14.1 / pandas 2.2.3 下完整重新生成：6,400 行的选择、释放与状态哈希逐行匹配，Brier、风险值和实现损失最大绝对差为零。早先聊天在不同 NumPy 环境中发现的候选池不匹配，不能用来断言这份提交源码无法复现。

可复现性通过仍需与协议有效性区分。提交版的在线评价提前用了下一轮才释放的探索标签；固定数量诊断有可行性缺陷；固定 ε 基线也未可靠达到自身目标。修复和证据详见审查报告。经验风险评分不提供现实数据同时覆盖证明，基线名称和消融归因也有解释边界。

## 运行

使用 Python 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python src/run_all.py --suite checks
```

公开测试不需要信用数据；缺少本地 NPZ 时，仅真实数据专属测试跳过，合成动态管线仍会运行。真实数据没有随本仓库再次分发，见 [数据说明](data/README.md)。

```bash
python scripts/verify_local_data.py /your/local/data/real_credit_datasets_authentic.npz
python src/run_all.py --suite p4 --smoke --data-dir /your/local/data --output-dir runs/smoke-001
python src/run_all.py --suite p4 --data-dir /your/local/data --output-dir runs/p4-001
python scripts/validate_p4.py runs/p4-001/p4_real_labels/p4_trajectories_10rounds.csv
```

输出目录必须是新目录。程序从当轮已释放标签训练模型，当前选中标签进入下一轮队列。正式 P4 为四数据集 × 16 seeds × 10 轮 × 10 方法；`--smoke` 仅 HMEQ、seed 4001。所有结果都记录来源和状态；不会恢复归档表来冒充新运行。

P3 可用 `--suite p3` 单独运行，保留最新版条件概率诊断。P0–P2 原始实现与冻结结果在归档目录，本次没有将其当作修复后的重新确认。统计以数据集内配对 seed 为回放随机性单位；不把轮次当独立样本，也不把四个旧队列重命名为新外部确认。

模型、采集、风险代理和求解范围的解释见 [方法与理论范围](docs/THEORY.md)。公开具名仓库不是匿名投稿补充材料；本仓库不声称完成投稿验收。
