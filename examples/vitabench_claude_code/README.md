# 实战：Claude Code CLI × VitaBench

端到端跑通一个真实学术基准：被测对象是 **Claude Code CLI**（完整 harness，含它自带的
上下文管理与工具环路），数据集是 **VitaBench 生活服务域**（多轮真实用户任务），
工具经 **MCP** 注入（被测对象的原生工具调用方式，控制变量干净）。

## 前置条件（你需要自己准备的）

1. **模型端点**：任意 OpenAI 兼容服务（vLLM / OpenRouter / 官方 API 均可）
2. **Claude Code CLI**：已安装并能对话——它需要一个 Anthropic 格式的模型端点。
   若你的模型是 OpenAI 格式，需要一个转换代理（开源方案任选，例如 claude-code-router
   或任意 anthropic-to-openai 代理），并在 `workdir/.claude/settings.local.json` 里配置：
   ```json
   {
     "env": {
       "ANTHROPIC_BASE_URL": "http://127.0.0.1:<你的代理端口>",
       "ANTHROPIC_AUTH_TOKEN": "<token>",
       "ANTHROPIC_MODEL": "<model-id>",
       "ANTHROPIC_SMALL_FAST_MODEL": "<model-id>"
     }
   }
   ```
3. **VitaBench 源码**（提供任务数据与其 Python 环境）：
   ```bash
   git clone --depth 1 https://github.com/meituan-longcat/vitabench /path/to/vitabench
   uv pip install -e /path/to/vitabench
   ```
4. 本框架已安装：`uv pip install -e ".[mcp]" --extra dev`（在仓库根目录）

## 配置 `.env`

```bash
# 被测模型（此处仅用于元信息记录与 openai-loop 基线对比）
AHEDD_AGENT_MODEL=<model-id>
AHEDD_AGENT_BASE_URL=<your-openai-compatible-endpoint>
AHEDD_AGENT_API_KEY=<key>

# 用户模拟器与判分器（vita 任务需要多轮对话与 rubric 判分）
AHEDD_USER_SIMULATOR_MODEL=<model-id>
AHEDD_USER_SIMULATOR_BASE_URL=<endpoint>
AHEDD_USER_SIMULATOR_API_KEY=<key>
AHEDD_JUDGE_MODEL=<model-id>
AHEDD_JUDGE_BASE_URL=<endpoint>
AHEDD_JUDGE_API_KEY=<key>

# Claude Code CLI 车道
AHEDD_CC_SSH=              # 留空 = claude CLI 在本机运行；填 user@host 则经 ssh 在远端运行
AHEDD_CC_DIR=/path/to/cc-workdir        # .claude/settings.local.json 所在目录
AHEDD_CC_NODE_BIN=<claude 可执行文件所在 bin 目录>
AHEDD_CC_PYTHON=python     # 远端模式下：已安装本框架的解释器（远端自起 MCP server 用）
```

## 运行（两种方式任选）

### 方式 A：脚本一键跑（推荐第一次使用）

```bash
cd examples/vitabench_claude_code
python run_experiment.py --cases 10711002
```

脚本做的事：拉起环境的 MCP server → 驱动 Claude Code（`--mcp-config` 原生工具调用）
→ 采集轨迹 → rubric 判分 → 生成 HTML 报告。每一步都有日志，出问题能定位到环节。

### 方式 B：CLI 分步跑（理解每一步）

```bash
cd <仓库根目录>

# ① 跑评测：MCP 原生工具模式 + 用户模拟器多轮对话
ahedd run --dataset vita --domain delivery --adapter claude-code \
    --tool-mode mcp --cases 10711002

# ② 离线判分（rubric 滑窗 judge + 轨迹动力学指标）
ahedd score --runs runs/delivery --dataset vita

# ③ HTML 诊断报告（轨迹回放 / rubric 红绿 / 环境终态 diff）
ahedd report --runs runs/delivery --out vita_report.html

# ④（可选）失败轨迹冻结为回归用例
ahedd freeze <run_id> --attribution tool.loop
```

### 用 openai-loop 基线做对照（同模型裸循环 vs 完整 harness）

```bash
ahedd run --dataset vita --domain delivery --adapter openai-loop --disable-thinking --cases 10711002
ahedd score --runs runs/delivery --dataset vita
```

`ahedd score` 末尾的套件汇总会给出两个车道的 pass_rate 与 token 消耗对比——
这就是最简单的 harness 消融实验：**同一模型、同一数据集、同一工具面，唯一变量是 harness**。

## 预期产物

| 产物 | 位置 |
|---|---|
| 轨迹（JSONL，先存后判） | `runs/delivery/<case_id>/<run_id>.jsonl` |
| 判分明细 + 过程指标 | 同目录 `<run_id>.score.json` |
| 环境终态 diff | 同目录 `<run_id>.envdiff.json` |
| HTML 诊断报告 | `vita_report.html` |
| 回归用例（freeze 后） | `regressions/cases/RC-xxxx.yaml` |

## 常见问题

- **MCP server 未就绪**：本地模式检查 8023 端口占用（`--mcp-port` 可换）；
  远端模式确认远端已 `pip install` 本框架且 `AHEDD_CC_PYTHON` 指向该解释器
- **claude 工具全被拒**：MCP 模式已内置 `--allowedTools mcp__ahedd`（server 级授权）；
  若仍被拒检查 claude 版本 ≥ 2.1
- **判分输出不是 JSON**：判分模型若是思考型，保持默认（自动关思考）；`--think` 慎用
