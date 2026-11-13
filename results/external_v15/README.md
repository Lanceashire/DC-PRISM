# external-v15：2026-11-13 L40 部分结果快照

`server_snapshot_20261113_161643.tar.gz` 是 2026-11-13 16:16:43（UTC+8）取得的运行中快照，包含 **99 / 384 条正式轨迹**、单独存放的本机旧 18 条、冻结程序和来源记录。它不包含约 840 MiB 的预提取特征；完整运行包另行通过 Release 附件分发。附件是否已经发布，以 Release 页面实际资产列表为准。

这 99 条来自 NVIDIA L40、PyTorch 2.7.1+cu128、双进程，覆盖网格 A 的部分种子；不能视为完整实验。程序仍在原服务器运行，所以此快照不是最新进度的实时视图。本次归档没有运行中途统计检验或得出新科学结论。

```bash
mkdir -p snapshot
tar -xzf server_snapshot_20261113_161643.tar.gz -C snapshot
```

关键路径：

- `snapshot/runs/main/`：99 条正式 CSV 和对应审计 JSON，每条 248 行，共 24,552 行。
- `snapshot/runs/main_local18/`：旧本机 18 条；包含与 L40 重复的种子，不应合并相加或用于默认统计。
- `snapshot/provenance/snapshot.json`：来源、计数、依赖、跨硬件续跑说明。
- `snapshot/provenance/snapshot_sha256.json`：所有快照原文件的 SHA-256。
- `snapshot/provenance/server/`：快照时的原服务器环境和运行状态记录。

程序、校验与迁移步骤见 [experiments/external_v15](../../experiments/external_v15/README.md)。保留全部 384 条后才能按冻结协议执行最终统计。

完整包名：`external_v15_resume_20261113.tar.gz`，881,346,281 字节（约 840.5 MiB）。SHA-256 见 `SHA256SUMS_full_package.txt`。本次归档提交已包含程序和小型结果快照，完整包的 Release 附件上传尚待本机 GitHub CLI 登录；没有附件时请勿把仓库源码归档当作完整数据包。
