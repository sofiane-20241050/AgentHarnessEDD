# cases/ —— 评测用例目录

用例两个来源：

- **数据集自带**：`DatasetProvider.load(domain)` 从数据集包内加载（如 vita 的 400 任务），不落在本目录。
- **用户自建 / 冻结回归**：`ahedd freeze <run_id>` 把失败轨迹冻结为回归用例，落在 `../regressions/cases/`。

本目录预留给手动编写的一次性用例（YAML 格式随 SPEC 文档敲定）。
