# examples/ —— 实战示例

以完整可跑的例子展示框架的典型用法（每个子目录自带 README 与运行脚本）。

## 计划中的示例

### `vitabench_mcp/` —— 学术基准经 MCP 供给任意 Agent

把 VitaBench 的域环境（工具 + 虚拟数据库 + 任务/rubric）包装为 `ahedd` 的
`DatasetProvider`，再以 MCP server 形式供给被测对象（Claude Code / DeepAgents /
任何 MCP 客户端），跑完整"run → score → 报告"流程。

学术 bench 的环境是 python 对象（非网络服务），通用 MCP 化需要逐域适配其工具实现——
因此按"数据集插件 + 示例流程"的形式放在 examples，而非核心包。

### `harness_ablation/` —— 同模型四车道消融

同一模型（vLLM 部署）、同一数据集（mock / vita）、四种 harness
（openai-loop / DeepAgents / tau / Claude Code），对比成功率、Step-to-Success
与 token 成本的帕累托曲线。

## 已内建的示例

mock 数据集（框架自测域）即内置示例：`ahedd run --dataset mock` + `ahedd score`，
参见根 README 的快速开始。
