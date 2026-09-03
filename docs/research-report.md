# Agent Harness 评测框架调研报告

> 项目代号：**AgentHarnessEDD**（Evaluation-Driven Development for Agent Harnesses）
> 日期：2026-09-03 ｜ 状态：v1.1（按评审意见修订：数据集插件化定位、rubric 资源收录、GT 与标注机制、通用性原则）
> 本报告是项目立项前的完整调研沉淀，作为后续 README / SPEC 文档的上游输入。

---

## 0. 摘要（TL;DR）

1. **定位**：做一个 Python 优先的 Agent Harness 评测框架，"评测 + 诊断 + 回归"三合一。方法论核心是 **EDD（Evaluation-Driven Development，评估驱动开发）**——先定义"怎么算成功"，再构建 Agent；每次失败都可归因、可复验、可沉淀为回归资产。
2. **场景与数据**：框架与数据集完全解耦——数据集以插件（`DatasetProvider`）形式接入，**数据形态适配框架抽象，而非框架适配数据**。首个接入插件为生活服务域的 VitaBench（ICLR 2026，MIT：66 工具 / 400 任务），其论文验证的 rubric 滑动窗口判分思路（人工一致性 κ=0.828）是判分通道设计的依据之一。coding 域（SWE-bench 型）与企业私有集走同一套抽象（见 §9–§11）。
3. **被测对象接入**：三车道架构——Python 原生 Harness（DeepAgents、huggingface/tau、通用 OpenAI 兼容循环）进程内集成；TS/Rust 系（DeepSeek Harness、Codex CLI、Pi）与闭源客户端（Claude Code）通过 **MCP / 插件 / RPC** 协议边界接入。
4. **差异化**：VitaBench/τ-bench 只出分；Inspect AI 要求在其 DSL 内写 Agent；LangSmith/Langfuse 是 SaaS 观测、无确定性环境回归。我们的生态位 = **确定性环境 + 轨迹原生诊断 + Bad Case→回归用例闭环 + 本地优先开源**。
5. **命名**：**AgentHarnessEDD**——仓库/项目名 `AgentHarnessEDD`，PyPI 包 `agentharness-edd`，导入名/CLI `ahedd`（占用均已实测核查，见 §7）。

---

## 1. 背景与定位

### 1.1 要解决的问题

Agent 落地团队的三个普遍痛点：

- **Bad Case 回流难**：失败案例散落在聊天记录、工单和口口相传中，无法结构化沉淀，更无法复验。"修复了吗？修好了吗？复发了吗？"三个问题都答不上来。
- **可观测性缺失**：只有一个总分，看不到失败发生在哪一环节（推理？工具选择？参数？交互澄清？），归因靠人肉翻日志。
- **指标体系缺失**：端到端成功率之外，缺少分环节、分维度的量化指标，无法对比"换模型 / 改 Prompt / 调工具描述"前后的差异。

### 1.2 方法论：EDD（评估驱动开发）

《AI Engineering》（Chip Huyen）提出的核心主张：**在动手搭 Agent 之前，先把"怎么算成功"定义清楚**。类比 TDD（测试驱动开发）：

| TDD（代码世界） | EDD（Agent 世界） |
|---|---|
| 先写测试再写实现 | 先定义评估标准（rubric/断言）再搭 Agent |
| 红测试 → 绿测试 | Bad Case 复现 → 修复通过 |
| 回归套件防复发 | Bad Case 冻结为回归用例，变更必跑 |
| 覆盖率 | Rubric 约束覆盖度 |
| CI 门禁 | Prompt/模型变更触发全量回归 |

成立的前提很现实：大模型是概率系统，同样输入两次运行可能不同（VitaBench 实测：温度设 0 仍会轨迹发散），无法像传统程序那样"读代码判对错"，唯一可依靠的是**可量化、可重复的评估体系**。EDD 就是本项目的第一性方法论，也是命名依据。

### 1.3 一句话定位

> **AgentHarnessEDD = 借鉴 Inspect AI、τ²-bench 等开源评测框架与《AI Engineering》方法论构想的"评测 · 诊断 · 回归"三合一框架：数据集与被测 Agent 皆插件化，轨迹原生，每次失败可归因、可复验、可沉淀为回归资产。**

---

## 2. VitaBench 深度调研（论文 arXiv:2509.26490 + 仓库 meituan-longcat/vitabench）

### 2.1 概况

- 来源：美团 LongCat 团队，ICLR 2026 接收；代码 MIT 协议，Python，架构从 **tau2-bench** 衍生（与本仓库 `benchmark/tau2-bench` 精读笔记同源，集成风险低）。
- 任务定位：生活服务场景（外卖配送 Delivery、店内消费 In-store、在线旅游 OTA、跨场景 Cross）的多轮交互式工具 Agent 评测。
- 难度现状：SOTA 模型跨场景 Avg@4 仅 **30.0%**，单场景也普遍低于 50%——说明该场景有充分区分度，做诊断类工具大有可为（分数低 = 失败样本多 = 诊断与沉淀的价值大）。

### 2.2 数据与环境统计（论文表 2）

| | Cross-Scen. | Delivery | In-store | OTA |
|---|---|---|---|---|
| 任务数 | 100 | 100 | 100 | 100 |
| API 工具总数 | **66**（写 27 / 读 33 / 通用 6） | 20 | 24 | 38 |
| 服务商 | 1,324 | 410 | 611 | 1,437 |
| 商品 | 6,946 | 788 | 3,277 | 9,693 |
| 交易记录 | 447 | 48 | 28 | 154 |

单个任务通常涉及 5–20 个服务商，部分任务超过 100 个候选商品；刻意混入**干扰项**（违反特定约束的候选项）制造大搜索空间、少有效解。交易历史支持"订和上次一样的餐食"这类消费模式推理。

### 2.3 形式化框架（论文 §3）

任务建模为 POMDP `(U, S, A, O, T, r)`：动作空间 = 工具调用 + 对话；状态 = 数据库状态 ⊗ 用户状态；转移函数分解为**确定性工具转移（Python 函数）+ 随机用户转移（LLM 模拟器）**。任务复杂度三维分解 `C = ⟨C_reason, C_tool, C_interact⟩`（推理/工具/交互）——这个三维框架直接映射到我们的**分环节指标体系**（§6.4）。

### 2.4 三方架构

```
被测 Agent（LLM + 工具） ⟷ 用户模拟器（LLM + 剧本 + 画像）
        │ tool_calls
        ▼
环境（66 个 Python 工具函数 + 虚拟数据库）
导演 = Orchestrator，驱动全场直到 ###STOP### 或失败
```

论文配置：用户模拟器 gpt-4.1，评估器 claude-3.7-sonnet（刻意与被测模型错开），温度 0.0，每任务跑 4 次。**在我们的框架中，三个 LLM 角色全部配置为 OpenAI 兼容端点**（vLLM 自部署 / OpenRouter / 官方 API），仅此一条即可覆盖其全部 LLM 依赖。

### 2.5 Rubric 滑动窗口评估器（论文 §3.3，核心资产）

- **设计**：每个任务人工设计原子化评分标准 `R = {r1..rk}`（如"距用户 500 米内的餐厅""用户仅食用素食"）。轨迹切成重叠窗口（w=10 轮，重叠 δ=2 轮），评估器维护 rubric 状态向量 `s ∈ {0,1}^k`——一旦某评分项在任意窗口被满足即永久标记；跨窗口向前传播保持判断连贯。
- **打分**：基准用全有或全无 `score = 1[Σsj = k]`，但**细粒度 rubric 天然提供稠密信号**（哪几项没满足），这正是我们要的分环节诊断数据。
- **可靠性（论文表 4 消融，GLM-4.5 跨场景轨迹）**：

| 配置 | Score | Task Acc. | Cohen's κ |
|---|---|---|---|
| 完整方法（rubric + 滑窗） | 20.0 | **95.0%** | **0.828** |
| 去 sliding window | 19.0 | 90.0% | 0.604 |
| 去 rubric checklist | 91.0（虚高） | 22.0% | 0.018 |
| 两者都去 | 82.0 | 32.0% | 0.067 |

结论：**rubric 结构是判分可用性的决定因素**（去掉后 κ 崩到 <0.07 且分数虚高到 82–91%），滑动窗口解决长轨迹超出上下文的问题。用户模拟器可靠性：信息保真度 9.48/10，人格一致性 9.34/10（100 对话双人标注）。
- **统计设计**：基于 32 次独立试验的重采样分析，k=4 次运行在精度与成本间最优（MSE 较 k=1 降低 77.5%，k=8 收益微小且开销翻倍）。

### 2.6 指标体系

- `Avg@4`：4 次运行的平均得分（主榜单指标）
- `Pass@4`：4 次中至少 1 次成功的概率（能力上限）
- `Pass^4`：4 次全部成功的概率（稳定性/可靠性，对生产更重要）
- 效率维度：论文实测思考型模型平均 23.8% vs 非思考 17.9%，且轮次更少（61.1 vs 69.9 轮）——轮次/token/成本是必采指标。

### 2.7 失效模式分析（论文 §6.3——Bad Case 归因分类学的起点）

对 Claude-4.1-Opus 跨场景失败 rubric 的人工归类：

| 错误类别 | 占比 | 典型表现 |
|---|---|---|
| **推理错误** | **61.8%** | 多约束复合目标处理失败、时空/常识推理系统性出错、选中干扰项 |
| **工具使用错误** | **21.1%** | 选错工具、参数传递失误、**无法从调用失败中恢复**（重复失败尝试而非换路径） |
| 交互错误 | 7.9% | 不主动澄清模糊需求、长对话丢失用户偏好 |
| 用户模拟器噪声 | 9.2% | 环境固有随机性，靠多次运行缓解（统计时剔除） |

三个反复出现的模式：① 时空推理与常识推理系统性弱；② **自我认知不足——有合适工具却放弃任务**；③ 失败恢复能力差——重复失败调用而非改变策略。这三条就是我们分诊看板上最高频的预期标签。

### 2.8 数据获取与工程形态

- 仓库：`github.com/meituan-longcat/vitabench`，`pip install -e .` 得到 `vita` CLI；`vita run --domain delivery --user-llm X --agent-llm Y --evaluator-llm Z`；支持 `--re-evaluate-file` 对已保存仿真离线重判（其"先存轨迹、后判分"的分离设计被我们采纳为框架原则）。数据集发布在 HuggingFace（README 提供链接），另有公开排行榜。
- 语言：任务数据以中文为主（源于真实中文生活服务平台），英文版在准备中。

### 2.9 适配方向：让数据适配框架（vita 插件），而非框架适配数据

**原则：不 vendor、不 fork、不抄 VitaBench（或任何基准）的代码。** VitaBench 以数据集插件（`VitaProvider`）的身份接入：其任务 JSON、域数据、rubric 清单在插件内被映射为框架的 `TaskCase` / `UserScenario` / `Environment` / `Scorer` 输入。框架核心抽象必须"包含"各基准的概念（表达力 ≥ 各基准），才能既装得下 VitaBench，也装得下 SWE-bench 型 coding 基准与企业私有集（§11）。

**vita 插件负责映射的东西**：任务（指令/剧本/rubrics）、域数据（服务商/商品/交易/干扰项）、66 个工具的 Python 实现封装、DB 写操作期望。

**框架侧借鉴（借鉴思想，不抄实现）**：
1. rubric 原子断言 + 滑动窗口长轨迹判分（κ=0.828 的设计依据）
2. 以 rubric 通道为主、确定性断言为辅——**不做终态哈希一刀切**（推荐/规划类行为不改变数据库但至关重要）
3. 先存轨迹、后判分的分离设计（判分器可换、可离线重跑）
4. 三 LLM 角色（被测/用户模拟器/判分器）的端点配置

**框架侧增强（差异化，任何单一基准都不提供）**：三车道被测接入、轨迹原生可观测与分环节指标、Bad Case→回归闭环、Pass^k 与效率成本维度。

### 2.10 企业私有测评集路线（预览，后续专题）

企业自建私有集的完整工作流是独立专题，此处只给轮廓：业务 API → env 工具的 schema 映射 → 种子数据脱敏/合成 → 剧本与 rubric 编写规范 → **GT Agent（带答案的考生）机器验收**（VitaBench 的做法：新任务先用 GT agent 跑，拿不到满分即题目有 bug；SABER 分析发现其 75+ 任务存在预期错误，靠 v1.0.1 版本化迭代收敛——私有集同样需要"数据质量本身被评估"的机制）→ 与公共集隔离的私有榜单。**当前阶段先全部用 VitaBench 公共数据。**

---

## 3. 被测 Harness 生态与接入矩阵

### 3.1 Python 原生（车道一：进程内集成，优先级最高）

**DeepAgents**（`langchain-ai/deepagents`）——旗舰被测对象
- LangChain 官方 "batteries-included agent harness"，Python，基于 LangGraph：规划工具（write_todos）、子 Agent 委派、文件系统上下文管理、可注入自定义工具/换模型/改 Prompt。
- 集成方式：`create_deep_agent(tools=[...])` 注入我们的环境工具；轨迹采集用 LangGraph middleware/callback 拦截，进程内零序列化开销，token 统计精确。

**tau**（`huggingface/tau`）——Pi 的官方 Python 移植，第二示例
- HuggingFace 出品的 Pi 极简编程 Agent Python 版，MIT，三层结构：`tau_ai`（多 Provider 含 OpenRouter/OpenAI 兼容端点）/ `tau_agent`（可移植大脑：messages/tools/events/loop/**harness**/sessions）/ `tau_coding`（CLI/TUI 前端）。
- **"Events are the contract"**：`async for event in harness.prompt(...)` 消费类型化事件流——与我们轨迹采集层的对接几乎是天然的；会话为持久化 JSONL，天然可回放。
- 工具即"带 schema 的普通类型化函数"，注入 VitaBench 工具无障碍。要求 Python 3.12+。

**通用 OpenAI 兼容 function-calling 循环**——兜底适配器
- 约 100 行的标准工具调用循环，直连任意 OpenAI 兼容端点（vLLM 自部署、OpenRouter、DeepSeek/GLM/Qwen 官方 API 全覆盖）。通用性最强，是各数据集开箱即测的"默认考生"与基线。

### 3.2 外部/异构（车道二/三：协议边界）

**DeepSeek Harness（dsh）**（`deepseek-ai/deepseek-harness`）
- DeepSeek 官方开源 Agent Harness，核心为 Node/TS（仓库含 `python/` 目录但主体是 pnpm + TS）。"Everything is a Plugin" 架构：模型适配器、工具、会话存储、权限、Agent 循环皆可插件替换。开源即现象级爆火（社区报道 24 小时破 5 万 Star）。
- 对我们的意义：① 其插件化思想值得借鉴（我们的适配器也做成插件式注册）；② 作为外部被测对象经协议边界接入。

**Codex CLI**（OpenAI，Rust/TS）——外部 CLI Agent，MCP 客户端，走协议边界。

**Pi**（`earendil-works/pi`，原 badlogic/pi-mono）——TS 极简编程 Agent；其 **RPC 模式（JSON over stdio 无头运行）** 是"子进程驱动外部 CLI Agent"车道的三车道参考实现。

**Claude Code**（Anthropic，闭源客户端）
- 两条集成路径，**MCP 为主**：① Claude Code 是成熟 MCP 客户端（`claude mcp add` 挂我们的环境 MCP Server，Agent 直接"住进"我们的仿真世界）；② 其 hooks/插件机制可作为补充（如失败上报钩子）。闭源不影响评测：我们在边界上观察工具调用与终态，不需要其内部实现。

### 3.3 接入矩阵总结

| 被测对象 | 语言 | 车道 | 集成方式 | 轨迹采集 |
|---|---|---|---|---|
| DeepAgents / LangGraph | Python | 进程内 | 工具注入 + middleware | callback 拦截（精确） |
| tau | Python | 进程内 | `AgentHarness` 事件流 | 事件流消费（原生契约） |
| 任意 OpenAI 兼容模型 | — | 进程内 | function-calling 循环 | 循环内录制（精确） |
| dsh / Codex CLI / Pi | TS/Rust | MCP / 子进程 RPC | MCP Server 或 stdio RPC | 协议消息录制 |
| Claude Code / Dify / Coze | 闭源 | MCP | 环境暴露为 MCP Server | MCP 消息录制 |
| 企业自研 Agent | 任意 | MCP / HTTP | 连接我们的环境端点 | 协议消息录制 |

**回答"插件/MCP 对 CLI 与客户端是否都好集成"：是。** CLI 型（Codex/Pi/dsh）普遍已支持 MCP 客户端或提供无头 RPC；客户端/平台型（Claude Code/Dify/Coze）原生支持 MCP outbound。MCP 是唯一同时覆盖"开源 CLI + 闭源客户端 + 自研服务"的通用面；框架原生插件（如 Claude Code hooks）作为锦上添花的补充通道。

---

### 3.5 模型访问层极简策略

LLM 访问层**只适配 OpenAI SDK 兼容端点**（AsyncOpenAI + base_url）：vLLM / OpenRouter / 各家官方 OpenAI 兼容 API 全覆盖，不做多 Provider 抽象——评测框架的重点在数据集抽象与轨迹诊断，不在模型接入层。

### 3.6 适配 2026 年代成熟 Harness 的内置模块

现代 Harness（DeepAgents、dsh、Claude Code 等）早已不是前几年"简单 ReAct 循环"的形态，普遍内置：规划（todo/plan 工具）、子 Agent（subagent 委派）、记忆/文件系统、权限系统。适配策略：

1. **双工具命名空间**：`env tools`（我们的仿真环境，框架注入并录制）+ `harness-native tools`（被测 Harness 自带，如 write_todos、subagent 派生）——后者不注入、不拦截执行，但其调用作为轨迹事件入轨，分诊时区分"规划行为"与"业务行为"。
2. **轨迹事件模型可扩展**：StepKind 不设封闭枚举，canonical 集合之外允许 harness 专属事件（plan / subagent / memory / handoff 等）。
3. **DeepAgents / tau 适配验收标准**：默认配置跑通同一数据集，planning 与 subagent 事件完整入轨，工具调用与 token 计量和 openai-loop 车道一致。

---

## 4. MCP 集成入门（答"到时候给我讲讲 MCP"）

**MCP（Model Context Protocol）= 工具协议的 USB-C。** 它标准化了"Agent（客户端）如何发现、调用外部工具/资源（服务器）"：服务器声明工具清单（名称/描述/JSON Schema 参数），客户端的模型决定何时调用，服务器执行并返回结构化结果。传输支持 stdio（本地子进程）与 HTTP/SSE（远程）。

**我们的用法——方向是反过来的：**
不是让我们的框架去连别人的工具，而是**把评测环境（66 个工具 + 虚拟数据库）包装成一个 MCP Server**。任何 MCP 客户端形态的被测 Agent（Claude Code、Codex CLI、dsh、Dify 工作流）连上来后，它"以为"自己在操作真实业务系统，实际上每次工具调用都发生在我们的沙箱里：可录制、可断言、可回滚。我们不集成框架——**框架来连我们的世界**。

**什么时候不用 MCP：** Python 原生被测对象（DeepAgents/tau）进程内注入工具更优——零序列化开销、无进程管理、token/延迟统计精确到调用。原则：**进程内 > MCP > 子进程 RPC**，按被测对象形态选车道，三者共用同一套环境内核与轨迹 Schema。

---

## 5. 竞品分析

| 竞品 | 强项 | 短板（相对本项目） |
|---|---|---|
| **Inspect AI**（UK AISI） | solver/scorer 抽象优雅、数据集与评测分离、Python | 要求 Agent 写在其 DSL 内；非轨迹原生；无环境状态断言与回归闭环 |
| **promptfoo / DeepEval** | 文本断言 DSL 上手快、CI 集成好 | 偏文本输出评测；工具调用/多轮环境/状态验证浅 |
| **tau2-bench / VitaBench** | 确定性环境 + DB/rubric 判分，学术权威 | 论文基准形态：只出分，被测形态单一，无可观测层、无回流机制 |
| **LangSmith / Langfuse** | 生产 trace 观测、SaaS 体验好 | 无确定性仿真环境、无终态断言回归；SaaS 优先、数据出境顾虑 |

**我们的生态位**：确定性环境内核（环境 + 各数据集插件）× 轨迹原生诊断（分环节指标 + HTML 报告）× Bad Case→回归闭环（EDD）× 本地优先开源（Python 核心 + MCP/插件广接入）。四者交集目前为空。

---

## 6. Bad Case 沉淀机制设计（核心章节）

### 6.1 为什么 Bad Case 难以沉淀（根因诊断）

1. **不可复现**：概率系统连温度 0 都会轨迹发散（VitaBench §6.1 实测），口述的失败经验无法验证"修没修好"。
2. **归因多维**：失败可能来自 Prompt/模型/工具描述/环境数据，交织在一起，人肉归因成本极高。
3. **无结构化载体**：散落在聊天记录、工单、群里，随时间蒸发。
4. **缺闭环验证**：修复后没有回归门禁，"按下葫芦浮起瓢"无法被系统性发现。

### 6.2 沉淀物形态：回归资产（Regression Asset）

每个被确认的 Bad Case 冻结为一条**版本化、可重放的资产**：

```yaml
# regressions/cases/RC-0007.yaml
id: RC-0007
source_run: runs/2026-09-10/delivery/task_0042   # 溯源到首次失败轨迹
domain: delivery
env_seed: 8842                                    # 数据库初始态 + 系统时间等
user_scenario: { persona: 急躁, known/unknown_info: ... }   # 用户剧本
rubrics: [500米内餐厅, 素食, 预算<80, ...]         # 原子断言
attribution:                                      # 归因标签（§6.4 分类学）
  primary: tool.param.value
  detail: "时间参数格式错误：传入 '今晚' 而非 ISO 时间"
baseline_trace: runs/2026-09-10/.../trace.jsonl   # 修复前基线（红）
created_by: human-confirmed                        # 自动分诊 + 人工确认
```

配套一份**修复前 trace 快照**作为"红测试"基线。资产进 `regressions/` 目录随 git 版本化——这就是"经验"的物理形态。

### 6.3 EDD 闭环（七步飞轮）

```
① 采集    全量任务运行，轨迹/状态/成本落盘（先存轨迹、后判分）
② 判分    rubric 滑窗 + 确定性断言，输出任务级 + rubric 级结果
③ 分诊    失败案例自动预打标（分类学 §6.4）+ 人工确认（LLM 预分类，人把关）
④ 资产化  一键冻结为回归用例（env_seed + 剧本 + rubric + 标签 + 红基线）
⑤ 门禁    prompt/模型/工具描述任何变更 → ahedd ci 全量回归，红绿一目了然
⑥ 看板    归因分布趋势：哪类错误占比升/降、集中在哪个域/哪个工具/哪个参数
⑦ 复验    修复后回归通过率 + 该类归因占比变化 → 量化"经验是否真的沉淀了"
```

**"错误如何沉淀为经验"的可度量答案：经验 = 通过回归门禁的那次 diff + 归因统计的下降曲线。** 不是 wiki 里的复盘文档，而是可重放的资产和可统计的趋势。

### 6.4 归因分类学（对齐论文 ⟨推理, 工具, 交互⟩ 三维，细化为可操作标签）

| 一级（论文） | 二级标签 | 判定线索（可自动/半自动） |
|---|---|---|
| 推理 (61.8%) | 约束遗漏 / 时空推理错 / 干扰项误选 / 放弃任务(自我认知) | 哪条 rubric 未满足 + 在哪一轮丢失；是否零写操作即终止 |
| 工具 (21.1%) | 工具选错 / 参数 schema 错 / 参数值语义错 / 失败不恢复 / 冗余调用 | 工具调用序列 vs 参考解；重复失败调用检测 |
| 交互 (7.9%) | 未澄清 / 澄清时机不当 / 丢失用户偏好 / 过早终止 | 澄清问答检测；偏好rubric在长对话后半段失效模式 |
| 噪声 (9.2%) | 用户模拟器错误 | 人工标记，统计剔除（同论文处理） |

### 6.5 看板如何反哺修复（示例推演）

- "参数值错误中 60% 集中在时间参数" → 修工具描述里的时间格式约定 + 补 few-shot → 跑回归集 → 该标签占比应下降
- "未澄清错误集中在 delivery 域" → 系统提示词补澄清策略 → 回归验证
- "失败不恢复" 标签在模型 A→B 切换后翻倍 → 换回或加恢复引导

每一条都是从"群里的抱怨"变成"门禁上的红绿"的路径。

---

## 7. 命名决策

### 7.1 定名：**AgentHarnessEDD**（三层命名法）

- 全称释义：**Agent Harness × Evaluation-Driven Development**——既标明赛道（Agent Harness 评测），又标明方法论（EDD，与《AI Engineering》主张同源）。"harness" 在 DeepSeek Harness 爆火后已成为行业术语，名字自带检索流量，看名知域。
- 代价是长（16 字符包名），用**三层命名法**解决——长名字负责"被发现"，短名字负责"被使用"：

| 层 | 名字 | 示例 |
|---|---|---|
| 项目 / 仓库名 | `AgentHarnessEDD` | GitHub 仓库、文档、对外引用 |
| PyPI 包名 | `agentharness-edd` | `pip install agentharness-edd` |
| 导入名 / CLI | `ahedd` | `import ahedd` / `ahedd run` / `ahedd ci` / `ahedd report` |

  （同 beautifulsoup4→bs4、scikit-learn→sklearn 的惯例。）
- 占用核查（2026-09-03 实测）：
  - PyPI：`agentharness-edd` ✅ 未占用（`agent-harness-edd` / `agentharnessedd` 变体亦空闲）
  - GitHub：仓库名按用户命名空间分配，无全局冲突，✅ 可建
  - 弃用记录：`edda`（PyPI 已占）、`agent-edd`（可用，被本更名取代）、`agent-edda`（GitHub 组织名已占）

### 7.2 GitHub About 文案（直接复制）

> **AgentHarnessEDD** — Evaluation-Driven Development for AI agent harnesses: trace-native evaluation, per-stage failure diagnosis, and a bad-case→regression flywheel. Python-first; MCP/plugin adapters for external harnesses. 像对待代码一样对待 Agent：先定义成功，再构建；每次失败都可归因、可复验、可沉淀。

### 7.3 口号

- EN: *Define success before you build. Diagnose every failure. Never regress twice.*
- CN: 先定义成功，再构建 Agent；每次失败，都可归因、可复验、可沉淀。

---

## 8. 下一步

1. ~~建 GitHub 仓库~~（已完成，`AgentHarnessEDD`）
2. 编写 README（已完成，双语版）与 SPEC 文档（轨迹 Schema、回归资产 Schema、适配器接口、判分通道、CLI 命令集，CLI 一律以 `ahedd` 为命令入口）
3. Demo 里程碑（D1–D4）：vita 数据集插件接入 + 通用 OpenAI 适配器跑通 delivery 10 任务 → DeepAgents/tau 适配器 + 轨迹落盘 → rubric 判分 + 分环节指标 + Pass^4 → HTML 诊断报告 + Bad Case 冻结演示
4. rubric 结构化格式定稿（对齐 Prometheus / promptfoo llm-rubric，见 §9）+ GT Agent 验收工具设计（见 §10）

## 9. 权威 rubric 与判分资源收录（2026-09 调研）

框架的 rubric 层不发明新轮子，格式对齐业界既有标准，方便互相迁移。

### 9.1 判分器与 rubric DSL

| 资源 | 是什么 | 对我们的落点 |
|---|---|---|
| [Prometheus 2 / prometheus-eval](https://github.com/prometheus-eval/prometheus-eval)（arXiv 2405.01535） | 开源判分专用模型：用户自定义 score rubric + 1–5 直接评估 / 两两对比，与人类判断一致性最高的开源 judge | rubric 的结构化格式（criteria + description + scale）作为 `TaskCase.rubrics` 的可选结构化形态；支持本地 judge（私有集数据不出域） |
| [promptfoo llm-rubric](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric/) | 自然语言 rubric 断言 DSL，另有 g-eval、agent-rubric（agentic 判分循环）变体 | YAML 用例中自然语言 rubric 的写法参考；agent-rubric 的"判分器也可带工具"思路借鉴到滑窗判分器 |
| [DeepEval G-Eval](https://deepeval.com/docs/metrics-llm-evals) | CoT 链式思考的自定义 criteria 判分（G-Eval 论文的工业实现） | judge prompt 的 CoT 结构参考 |
| Inspect AI scorers | solver/scorer 分离 + 多种 scorer 组合模式 | 判分通道的组合与注册方式参考 |

### 9.2 基准侧 Ground Truth 形态（GT 不止一种）

| 基准 | GT 形态 | 启示 |
|---|---|---|
| SWE-bench | fail-to-pass 单元测试 + pass-to-pass 回归测试（完全确定性） | coding 域接入方式：测试套件 = 纯确定性 Scorer 通道（§11 映射表） |
| WebArena | 功能性检查（URL / 页面元素 must_include 等） | 网页域确定性断言的形态 |
| OSWorld | 每任务 setup / evaluator 脚本 | "环境脚本化验收"模式 |
| τ²-bench | DB 写操作断言 + ACTION 参考解（仅对照，非唯一解） | 参考解不作判分依据，是教训 |
| VitaBench | rubric 原子清单 + 滑窗状态向量 | rubric 为主通道的依据（κ=0.828 vs 无 rubric κ<0.07） |

### 9.3 失效分类学（Bad Case 分诊标签体系的上游）

| 资源 | 内容 |
|---|---|
| [MAST（arXiv 2503.13657，UC Berkeley，NeurIPS 2025）](https://github.com/multi-agent-systems-failure-taxonomy/MAST) | 首个多 Agent 系统失效分类学：3 大类 14 种失效模式 + 大规模人工标注集 MAST-Data；并给出多 Agent 场景"GT 难以核验"时的标注方法论 |
| VitaBench §6.3 | 单 Agent 工具任务的失效分布：推理 61.8% / 工具 21.1% / 交互 7.9% / 噪声 9.2% |

我们的归因分类学（§6.4）= VitaBench 三维框架打底 + MAST 失效模式作多 Agent 扩展位（DeepAgents 子 Agent 场景分诊用）。

## 10. Ground Truth 与标注机制（答：需要 GT 吗？谁来标？）

**需要，但 GT ≠ 唯一参考解；GT = 断言集合。** 三种形态：

1. **断言型 GT（主体）**：rubric 原子清单（行为目标）/ 测试套件（coding 域）/ DB 写操作期望（状态类）。特征：可判定、多条相互独立、允许殊途同归——Agent 走任何能满足全部断言的路径都算过（过程自由、结果零容忍）。
2. **参考解轨迹（辅助）**：如 τ² 的 ACTION 通道、VitaBench 的 actions 字段。只作对照与归因参考，**不作判分依据**——把它当唯一解是"强假设"，τ² 社区因此大多不启用该通道。
3. **判分器（不是 GT）**：LLM judge / GT Agent 都是"GT 的执行者或代理"，不是真值本身。judge 的可靠性来自 rubric 结构（κ=0.828 vs 无 rubric κ<0.07），而非模型本身。

**谁来标注：**

- **公共集**：数据集作者人工标注（VitaBench 的 rubric 即从任务信息人工提取的原子准则），再用 **GT Agent（带答案的考生）机器验收**——新任务 GT Agent 拿不到满分即题目有 bug；即便如此 SABER 分析仍发现 VitaBench 75+ 任务预期错误，靠版本化迭代收敛（v1.0.1 前后分数不可比）——**数据质量本身也是被评估的对象**。
- **私有集**：业务专家按 rubric 编写规范标注 → GT Agent 验收 → 版本化。框架侧提供：rubric 编写模板与 lint（一条 rubric 只表达一个可判定约束）、GT Agent 验收工具、私有/公共集隔离跑分（后续专题）。
- **统计可靠性**：k=4 次独立运行（温度 0 仍会轨迹发散；MSE 较 k=1 降低 77.5%）。

## 11. 通用性设计原则：抽象表达力 ≥ 各基准（含 coding 域映射示例）

基类"包含"目标基准的概念，是数据集插件化的前提。同一套抽象在不同域的映射：

| 概念 | 通用抽象 | VitaBench 映射 | SWE-bench 型映射 |
|---|---|---|---|
| 任务 | `TaskCase` | 指令 + 剧本 + rubrics | issue 描述 + repo 快照 + 测试清单 |
| 用户侧 | `UserScenario`（可空） | LLM 用户模拟器（剧本+画像） | 无（单轮指令，user_simulator=None） |
| 环境 | `Environment` | 66 工具 + 虚拟 DB（快照/diff） | docker 沙箱 + shell/fs 工具（镜像即 seed） |
| 断言 | `Scorer` 双通道 | rubric 滑窗 + 写操作校验 | fail-to-pass 测试套件（纯确定性）+ 可选 rubric（行为/效率） |
| 轨迹 | `Trajectory` | 对话 + 工具调用 | patch + 命令历史 + 测试输出 |

coding 域的 docker 沙箱以可选依赖（`agentharness-edd[swe]`）形式提供，核心抽象不感知 docker。前几年"Agent 评测 ≈ 简单 ReAct 循环"时代的产物（WebArena/OSWorld 的脆弱性、SWE-bench 的数百 GB 镜像）正是要绕开的坑：**核心轻量，重资产全部插件化**。

## 参考资料

- VitaBench 论文：arXiv:2509.26490（中文版 PDF 已存本地，全文抽取于 `E:\Downloads\vitabench_paper2.txt`）
- VitaBench 仓库：https://github.com/meituan-longcat/vitabench
- τ²-bench 精读笔记：本仓库 `benchmark/tau2-bench/tau2-bench-设计思路与测评方式总结.md`
- DeepAgents：https://github.com/langchain-ai/deepagents ｜ https://docs.langchain.com/oss/python/deepagents/overview
- DeepSeek Harness：https://github.com/deepseek-ai/deepseek-harness
- Pi：https://github.com/earendil-works/pi （原 badlogic/pi-mono）｜ RPC 模式文档见其 `packages/coding-agent/docs/rpc.md`
- tau（Python）：https://github.com/huggingface/tau
- Prometheus 2 / prometheus-eval：https://github.com/prometheus-eval/prometheus-eval ｜ 论文 arXiv:2405.01535
- MAST（多 Agent 失效分类学）：https://github.com/multi-agent-systems-failure-taxonomy/MAST ｜ 论文 arXiv:2503.13657
- promptfoo llm-rubric：https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric/
- DeepEval G-Eval：https://deepeval.com/docs/metrics-llm-evals
