# external-v15 服务器运行说明

外部有效性实验：在 8 个标准多分类数据集上（视觉：CIFAR-10、CIFAR-100 粗类、Fashion-MNIST、SVHN；文本：AG News、DBpedia-14、IMDB、Yelp），用冻结的预训练编码器加重训的 softmax 头，沿完整学习曲线比较三种 oracle（测试集选样、验证集选样、事前选样）和 5 个策略。共 384 条学习曲线：

| 网格 | 编码器 | 学习头 | 种子 | 条数 |
|---|---|---|---|---|
| A | DINOv2-small / BGE-small | MLP | 191001–191024 | 192 |
| B | MobileNetV3 / MiniLM | MLP | 192001–192012 | 96 |
| C | DINOv2-small / BGE-small | 线性 | 193001–193012 | 96 |

特征已提好（`features/`，840 MB），服务器上**不需要联网下载数据或模型**，只需要安装 Python 依赖。

## 1. 上传与安装

```bash
# 本机打包后上传（或直接 rsync 整个目录）
rsync -avP external_v15_deploy/ user@server:~/external_v15/
ssh user@server
cd ~/external_v15
bash setup.sh system     # 推荐：直接用镜像自带的 PyTorch；若镜像没有 PyTorch，改用 bash setup.sh cu124
```
`setup.sh` 最后会自动运行 `check_env.py`：检查 GPU、核对所有源码和特征文件的哈希是否与冻结时一致、训练一个小模型。看到 `PASS` 再继续。需要 Python 3.10 以上。推荐镜像：PyTorch 2.5+ / Python 3.12 / CUDA 12.x / Ubuntu 22.04。

## 2. 运行

```bash
bash run_server.sh single 3     # 单卡：3 个进程共用一块 GPU（显存 ≥12 GB 可试 4–6）
bash run_server.sh multi 3      # 三卡及以上：A/B/C 三组分别放在 GPU0/1/2 上并行
bash run_server.sh status       # 随时查看进度
```
- 程序后台运行（nohup），断开 SSH 不受影响；日志在 `runs/main*.log`。
- **可断点续跑**：中断后再执行同一条命令，已完成的轨迹会自动跳过。
- 本机 RTX 5060 Laptop 上每条约 106 秒（3 进程）。A100/4090 级别预计 2–4 倍速：单卡约 3–6 小时，三卡约 1.5–2.5 小时。

## 3. 完成后统计

```bash
bash run_server.sh stats
```
结果在 `results/`：`prediction_check.json`（5 条登记预测 Q1–Q5 是否成立）、各网格的汇总与检验表、`rows.csv.gz`（全部逐步结果）。把整个 `results/` 和 `runs/` 拷回本机即可，我来写报告。

## 4. 关于本机已完成的 18 条

`runs/main_local18/` 是本机 RTX 5060 上已完成的 18 条（网格 A 的前 3 个种子）。默认**不使用**，服务器上全部 384 条重算，以保证所有结果来自同一硬件（开启了 TF32，不同 GPU 上数值不会逐位相同）。若想省时间，开跑前执行 `mkdir -p runs/main && cp runs/main_local18/*.csv runs/main/` 即可续用。

## 5. 流程记录（如实披露）

- 协议、源码和特征在任何正式种子运行前冻结（工作目录 git 提交 `c69fb8a`，2026-11-13 00:58）。
- 2026-11-13 00:59–01:31 本机跑完 18 条后会话中断。
- 2026-11-13 10:32 为验证部署包的统计合并功能，用这 18 条部分结果运行过一次 `stats_v15.py`，输出中显示了一个部分数据上的描述性数值（Spearman −0.14）。这是一次中途查看；此后协议、代码、种子和样本量均未作任何改动，该数值不进入任何结论。
