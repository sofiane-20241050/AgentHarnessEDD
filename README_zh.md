<div align="center">

**简体中文** ｜ [English](README.md)

# AgentHarnessEDD

**面向 Agent Harness 的评估驱动开发（EDD）框架**

评测 · 诊断 · 回归 —— 先定义"怎么算成功"，再构建 Agent。

`pip install agentharness-edd` ｜ 导入名 / CLI：`ahedd`

🚧 脚手架阶段（v0.1.0）：基类契约已定，运行时逐步填充

</div>

---

## 为什么需要它

Agent 是概率系统：同样输入两次运行可能给出不同结果，你无法像读代码那样"读一遍就知道它对不对"。团队真正反复遇到的三个问题：

1. **Agent 考了多少分？** —— 换个模型、改段 Prompt，到底变好还是变坏？
2. **错在哪一环？** —— 推理没跟上？工具选错了？参数传错了？还是没向用户澄清？
3. **修完之后复发了吗？** —— Bad Case 修一个丢一个，没有回归防线。

AgentHarnessEDD 用 **EDD（Evaluation-Driven Development）** 回应：像 TDD 之于代码一样对待 Agent——先定义成功标准，再构建；每次失败都可**归因**、可**复验**、可**沉淀**为回归用例。

---

## 一次评测由哪些组件构成

这是本项目的"基类地图"。**接入任何一套评测数据集（公开基准或企业私有集），本质上是提供以下六类组件**——每一类都有对应基类与注册机制，全部可插拔：

| # | 组件 | 基类 / 模块 | 你要提供什么 |
|---|------|------------|-------------|
| 1 | **数据集** | `ahedd.datasets.DatasetProvider` | 任务集：初始指令 + 用户剧本 + rubric 断言（+ 可选环境种子） |
| 2 | **环境** | `ahedd.env.Environment` | 工具集 + 虚拟数据库；确定性内核，支持快照 / diff |
| 3 | **被测 Agent** | `ahedd.adapters.AgentAdapter` | 见下方"三车道"——进程内适配器 / MCP / RPC |
| 4 | **LLM 端点** | `ahedd.config.ModelSpec` | 角色化端点配置：被测模型、用户模拟器、判分器（三个角色，全部 OpenAI 兼容：vLLM / OpenRouter / 各家官方 API） |
| 5 | **判分** | `ahedd.scoring.Scorer` | 双通道：rubric 滑动窗口 LLM 判分 + 确定性断言（写操作 / 轨迹规则） |
| 6 | **指标** | `ahedd.scoring.metrics` | Avg@k / Pass@k / Pass^k + 分环节指标（推理 / 工具 / 交互）+ 效率（轮次 / token / 成本） |

配套的两个横切层：

- **轨迹（`ahedd.trace`）**：统一轨迹 Schema（JSONL 落盘，先存轨迹、后判分——判分器可换可重跑）
- **回流（`ahedd.regression`）**：失败轨迹一键冻结为回归用例（`RC-xxxx.yaml`），CI 门禁防复发

## 接入一套评测数据集的完整流程

以首个适配的评测数据集 **VitaBench**（生活服务域：外卖配送 / 店内消费 / OTA / 跨场景，400 任务、66 工具，ICLR 2026）为例：

```text
① 实现数据集接入     class VitaProvider(DatasetProvider)   # 任务、剧本、rubric 断言
② 挂接仿真环境       provider.build_environment(domain)    # 工具 + 虚拟数据库（含干扰项数据）
③ 配置三个 LLM 角色  models.yaml: agent / user_simulator / judge
④ 选择被测车道       ahedd run --dataset vita --adapter openai-loop --model <被测模型>
⑤ 出报告             ahedd report                         # 分数 + 分环节诊断 + 轨迹回放
⑥ 沉淀失败           ahedd freeze <run_id>                # Bad Case → 回归用例 → ahedd ci
```

> 数据集不绑定任何单一基准：VitaBench 只是 `ahedd` 的第一个数据集插件。企业私有集按同一套基类（`DatasetProvider` + `Environment`）自建即可，公共集与私有集隔离跑分。

## 被测 Agent 的三车道接入

| 车道 | 适用对象 | 集成方式 | 轨迹采集 |
|------|---------|---------|---------|
| **进程内** | Python 系：DeepAgents（LangGraph）、tau、任意 OpenAI 兼容模型 | 工具注入 + 事件流/回调 | 精确（token 级） |
| **MCP** | 闭源客户端：Claude Code、Codex CLI、Dify、Coze 等 | 环境暴露为 MCP Server，Agent 主动连接 | 协议消息录制 |
| **RPC** | CLI 型 Agent（无头模式） | stdio JSON-RPC 子进程驱动 | 协议消息录制 |

原则：**进程内 > MCP > RPC**，按被测对象形态选车道；三条车道共用同一套环境内核与轨迹 Schema。

## 快速开始

```bash
uv venv .venv
uv pip install -e ".[dev,deepagents,tau]"

ahedd datasets list    # 已注册数据集（当前内置 mock 自测域）
ahedd adapters list    # 已注册适配器（openai-loop / deepagents / tau）

# 配置 .env（参考 .env.example）后，用真实模型跑 mock 域：
ahedd run --dataset mock --adapter openai-loop --disable-thinking
ahedd run --dataset mock --adapter deepagents --disable-thinking
ahedd run --dataset mock --adapter tau
```

## 目录结构

```text
AgentHarnessEDD/
├── src/ahedd/
│   ├── datasets/     # 数据集接入基类 + 注册机制（任务/rubric/剧本）
│   ├── env/          # 仿真环境基类（工具 + 虚拟数据库 + 快照/diff）
│   ├── adapters/     # 被测 Agent 三车道适配器
│   ├── trace/        # 统一轨迹 Schema 与录制器
│   ├── scoring/      # 判分双通道 + 指标（Avg@k / Pass@k / Pass^k）
│   ├── regression/   # Bad Case → 回归用例资产
│   ├── report/       # 单文件 HTML 诊断报告
│   ├── config.py     # models.yaml：三角色 LLM 端点配置
│   ├── runner.py     # 单任务评测编排（采集→落盘→判分分离）
│   └── cli.py        # ahedd 命令入口
├── cases/            # 用例（数据集自带 / 用户自建）
├── regressions/      # 冻结的回归用例资产（随 git 版本化）
├── docs/             # 调研报告等文档
└── tests/
```

## 路线图

- [ ] **D1** 环境与数据集接入：`vita` 数据集插件（openai-loop 适配器与 mock 自测域已跑通真模型）
- [x] **D2** 第二/三被测车道：DeepAgents、tau 进程内适配器（已接真模型验证）；轨迹 JSONL 落盘
- [x] **D3** 判分双通道 + 指标：rubric 滑窗 judge（真模型验证，可抓 tau 拒改址类 bad case）、确定性断言、轨迹动力学指标（有效动作比/错误恢复/STS/帕累托）、`ahedd score` 离线判分
- [ ] **D4** HTML 诊断报告 + `ahedd freeze` 回归冻结演示
- [ ] 数据集插件化（entry-points）、企业私有集工作流指南

## 文档

- [调研报告：定位、VitaBench 深度调研、竞品分析、Bad Case 沉淀机制设计](docs/research-report.md)

## 许可证

[Apache-2.0](LICENSE)
