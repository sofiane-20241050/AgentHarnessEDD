# examples/ —— 可运行示例

| 示例 | 内容 |
|---|---|
| [`vitabench_claude_code/`](vitabench_claude_code/) | 完整实战：用 Claude Code CLI + 任意 OpenAI 兼容模型跑 VitaBench 真实任务，出分出报告 |
| [`custom_dataset/`](custom_dataset/) | 15 分钟接入你自己的业务数据集（Provider + Environment + 任务） |
| [`custom_adapter/`](custom_adapter/) | 15 分钟接入你自己的 Agent Harness 作为被测对象 |

## 环境准备（所有示例通用）

```bash
git clone https://github.com/sofiane-20241050/AgentHarnessEDD && cd AgentHarnessEDD
uv venv .venv && uv pip install -e ".[dev,mcp]"
cp .env.example .env        # 填入你自己的模型端点（任意 OpenAI 兼容服务）
```

框架本身不绑定任何模型服务商：被测模型、用户模拟器、判分器三个角色都在 `.env` 里
用 `AHEDD_{AGENT|USER_SIMULATOR|JUDGE}_{MODEL,BASE_URL,API_KEY}` 独立配置。
