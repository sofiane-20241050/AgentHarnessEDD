# AgentHarnessEDD 使用指南

> 从零到出报告的完整操作手册。所有命令在项目根目录的 PowerShell 或 bash 中执行。

## 前置准备（一次性）

```bash
# 1. 安装
git clone https://github.com/sofiane-20241050/AgentHarnessEDD && cd AgentHarnessEDD
uv venv .venv && uv pip install -e ".[dev,mcp]"

# 2. 配置模型端点（任意 OpenAI 兼容服务）
cp .env.example .env
# 编辑 .env，填入三角色端点：
#   AHEDD_AGENT_*          被测模型（考生）
#   AHEDD_USER_SIMULATOR_* 用户模拟器（多轮对话数据集需要）
#   AHEDD_JUDGE_*          判分器（rubric 评估）

# 3.（可选）安装 VitaBench 数据集
git clone --depth 1 https://github.com/meituan-longcat/vitabench /path/to/vitabench
uv pip install -e /path/to/vitabench
```

## 核心 CLI 命令

### `ahedd run` — 跑评测

```bash
ahedd run --dataset <名> --domain <域> --adapter <车道> [选项]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | `mock` | 数据集名（`ahedd datasets list` 查看） |
| `--domain` | 域列表第一个 | 域（如 `delivery` / `instore` / `ota`） |
| `--cases` | 全量 | 逗号分隔的 case id |
| `--adapter` | `openai-loop` | 被测车道（`ahedd adapters list` 查看） |
| `--tool-mode` | `mcp` | 工具注入：`mcp`=原生 MCP（CC 官方通道）/ `text`=JSON 块兜底 |
| `--trials` | `1` | 每任务独立运行次数 k（Pass^k 采样） |
| `--concurrency` | `1` | 并发上限（MCP 模式强制串行） |
| `--max-tokens` | `4096` | 单次补全最大 token |
| `--disable-thinking` | 关 | vLLM 思考模型关思考（提速） |
| `--no-user-sim` | 关 | 禁用用户模拟器（单轮直跑） |
| `--mcp-port` | `8023` | MCP server 端口 |
| `--cc-timeout` | `900` | claude-code 车道子进程超时（秒） |

**四个车道速查**：

| 车道 | 命令 | 特点 |
|------|------|------|
| `openai-loop` | `ahedd run --dataset mock` | 裸函数调用循环（基线），无 MCP 基建 |
| `deepagents` | `ahedd run --dataset mock --adapter deepagents` | LangGraph 完整 harness（planning/subagent） |
| `tau` | `ahedd run --dataset mock --adapter tau` | HuggingFace tau 极简 harness（事件流） |
| `claude-code` | `ahedd run --dataset mock --adapter claude-code --tool-mode mcp` | Claude Code CLI（MCP 原生工具） |

### `ahedd score` — 离线判分

```bash
ahedd score --runs runs/<域> --dataset <名> [选项]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--no-judge` | 关 | 跳过 LLM 判分，只出确定性指标 |
| `--think` | 关 | 判分模型开思考（慎用，可能破坏 JSON） |
| `--strict-state` | 关 | **强制开启终态断言**（所有数据集） |
| `--no-state-check` | 关 | **强制关闭终态断言** |

**终态断言策略**（`expected_states` 字段级 DB 比对）：
- **vita 数据集默认关闭**——对齐 VitaBench 论文口径（官方 evaluator 不消费 final_state）
- **自定义数据集默认开启**——expected_states 由数据集作者自行定义，属增强保障
- 显式 `--strict-state` / `--no-state-check` 覆盖默认

### `ahedd report` — HTML 诊断报告

```bash
ahedd report --runs runs/<域> --out report.html
```

生成零依赖单文件 HTML：轨迹时间轴回放（工具调用/思考链/错误着色）、rubric 红绿 chip、环境终态 diff、套件汇总。

### `ahedd freeze` — 失败轨迹冻结为回归用例

```bash
ahedd freeze <run_id> [--attribution <标签>]
```

| 归因标签 | 含义 |
|----------|------|
| `tool.loop` | 同工具死循环 |
| `tool.param` | 参数错误 |
| `reasoning.constraint` | 约束遗漏/干扰项误选 |
| `reasoning.unfinished` | 超轮次未完成 |

产物：`regressions/cases/RC-xxxx.yaml` + 基线轨迹副本 `regressions/traces/`。

### `ahedd mcp serve` — 手动起 MCP server

```bash
ahedd mcp serve --dataset mock --port 8023 --events-file events.jsonl [--stdio]
```

通常不需要手动——`ahedd run --tool-mode mcp` 自动拉起/按 case 重启。

## Bad Case 回流（核心价值）

### 成功轨迹的用途

| 用途 | 操作 |
|------|------|
| **基线对比** | 换模型/改 Prompt 后重跑，对比 pass_rate 与 token 消耗 |
| **能力毕业** | 能力集（低通过率）中的稳定成功项移入回归集（接近 100%） |
| **few-shot 素材** | 成功轨迹的推理/工具选择模式可提取为 Prompt few-shot |
| **效率参照** | STS 分布（成功轨迹的步数中位数）标定"多少步算正常" |

### 失败轨迹的用途（EDD 飞轮）

```bash
# ① 跑评测 → 失败轨迹自动落盘 runs/<域>/<case_id>/<run_id>.jsonl
ahedd run --dataset vita --domain delivery --cases 10711001

# ② 判分 → 得知哪些 rubric 未满足、有无违例
ahedd score --runs runs/delivery --dataset vita

# ③ 诊断 → 打开 HTML 报告看失败发生在哪一步
ahedd report --runs runs/delivery

# ④ 冻结 → 把失败轨迹变成永久回归用例（防复发）
ahedd freeze <run_id> --attribution tool.loop

# ⑤ 修复 → 改 Prompt / 换模型 / 修工具描述

# ⑥ 复验 → 重跑全量回归，确认修好了且没引入新问题
ahedd run --dataset mock --cases <之前冻结的所有case_id>
ahedd score --runs runs/mock --dataset mock
```

**沉淀公式**：经验 = 通过回归门禁的那次 diff + 归因统计的下降曲线。
不是 wiki 里的复盘文档，而是可重放的资产（RC-xxxx.yaml）和可统计的趋势（归因标签分布）。

## 自定义数据集（15 分钟）

参见 `examples/custom_dataset/my_provider.py`——实现 `DatasetProvider` + `Environment` 契约，import 后 `ahedd datasets list` 出现即注册。

## 自定义 Harness（15 分钟）

参见 `examples/custom_adapter/my_adapter.py`——实现 `AgentAdapter` 契约（`name` + `run()`），注册后 `ahedd adapters list` 出现。
