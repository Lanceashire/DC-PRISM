# external-v15：可迁移运行包与部分结果（2026-11-13）

本包对应 DC-PRISM 的 external-v15 扩展实验。入口为 `run_v15.py`，训练头在 `heads.py`，冻结方案在 `protocol.json` / `freeze.json`，最终统计入口为 `stats_v15.py`。研究源码、参数和种子保持冻结版本，新增脚本只负责部署、后台监管和文件验证。

## 本次新运行的进度

- 快照时间：**2026-11-13 16:16:43 +08:00**。
- 正式结果：**99 / 384 条（25.78%）**，来自 NVIDIA L40、PyTorch 2.7.1+cu128、双进程。
- 当前是 **PARTIAL_RUNNING_SNAPSHOT**。服务器取得快照后仍在运行，本包不包含当时正在计算、尚未完整保存的轨迹。
- 每条含 31 个时间点（初始化 + 30 轮）× 8 个方法，共 **248 行**；99 条共 **24,552 行**，均已检查结构、有限数值、登记种子与审计文件。
- CIFAR-10、CIFAR-100 粗类、Fashion-MNIST 各 13 条；SVHN、AG News、DBpedia、IMDB、Yelp 各 12 条。本快照只覆盖网格 A 的部分种子，B/C 尚未完成。
- `runs/main_local18/` 另存本机 RTX 5060 的旧 18 条，**不计入这 99 条，不参与默认续跑或正式统计**。与服务器结果存在重复种子，不能直接相加。
- 此次发布只检查进度与文件完整性，没有运行部分数据上的假设检验或生成中途结论。原服务器说明中已披露较早对本机 18 条的一次中途查看；该历史记录保留。

## 数据内容与实验范围

`features/` 包含 8 个标准数据集、两套编码器的 **16 个预提取特征 NPZ**（约 840 MiB），并含标签与训练/测试划分。它们是本程序直接使用的数据，不是原始图片/文本全量数据，也不包含模型权重。特征文件的 SHA-256 在 `freeze.json` 中，来源及版本在 `assets_manifest.json` 中。安装依赖后，运行实验无需再次下载原始数据或编码器模型。

总计划为 384 条：A=8 数据集×24 种子，B=8×12，C=8×12。实验范围为冻结编码器加重训分类头，不涉及端到端微调。`cifar100c` 在这里指 CIFAR-100 的 20 个粗类别。

## 在另一台 Linux / NVIDIA 服务器续跑

下载完整 Release 附件 `external_v15_resume_20261113.tar.gz` 和 `SHA256SUMS.txt`。仅克隆 GitHub 仓库不会获得特征数据；仓库中的源码及独立结果快照用于查阅与审计。

```bash
sha256sum -c SHA256SUMS.txt
tar -xzf external_v15_resume_20261113.tar.gz
cd external_v15_resume_20261113
bash setup.sh system
bash resume.sh 2
bash status.sh
```

推荐使用带 CUDA PyTorch 的镜像，`setup.sh system` 复用镜像中的 PyTorch。`requirements_server_snapshot.txt` 记录原服务器的实际版本；原 `requirements.txt` 记录较早本机版本，二者不同，勿把它当作原服务器环境。若镜像没有 PyTorch，按 `setup.sh` 的 CUDA 安装分支准备环境后再运行。

默认在 `runs/main/` 继续，程序自动跳过 99 条已完成轨迹，计划还需 **285 条**。后台监管与 SSH 会话分离，关闭终端不影响计算。全部 384 条完成、逐文件完整性校验通过后，自动运行冻结的 `stats_v15.py`，输出至 `results/`；出错会记入 `runs/supervisor_state.json` 和日志，不会把失败标为完成。`status.sh` 同时检查进程存活，避免把残留 `RUNNING` 当成正在运行。

**跨硬件续跑应如实标注混合硬件来源。** 本实验启用了 TF32；不同 GPU / 库版本不能保证逐位相同。`provenance/snapshot.json` 记录这 99 条的来源，`resume.sh` 为后续机器记录 GPU、环境和启动时已完成的文件。如果希望全部结果来自新机器，可先把 `runs/main` 改名保留，再建立空的 `runs/main` 后启动；这样会重算 384 条，而非续跑 285 条。

```bash
mv runs/main runs/l40_snapshot_99
mkdir -p runs/main
bash resume.sh 2
```

## 耗时说明

原 L40 双进程在约 4 小时 45 分钟完成 98 条，等效约 **20.6 条/小时**；按这一早期平均速度，完整 384 条约 18.6 小时。此估计混合了首批数据集，后续网格包含不同编码器与线性头，不能视为固定速度或新机器的承诺。原 README 中关于其他 GPU 的预估没有在本次运行中验证。

## 校验与来源

- `python verify_snapshot.py`：校验已归档文件哈希与所有现有正式 CSV 的完整结构，不计算统计结论。
- `venv/bin/python check_env.py`：校验全部冻结源码、协议和特征哈希，并做 GPU 小模型检查。
- `provenance/server/`：快照时的环境、GPU、进度记录。这里的 RUNNING 属于原服务器，不代表新机器进程。
- `provenance/snapshot_sha256.json`：归档原文件的 SHA-256。冻结研究文件的原始哈希仍以 `freeze.json` 为准。
- 代码仓库：<https://github.com/Lanceashire/DC-PRISM>。

数据来源及模型的原有使用条件仍适用；本包未改变其归属。
