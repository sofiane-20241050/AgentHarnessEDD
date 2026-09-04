# Rubric 与判分设计调研（开源项目横评）

> 2026-09 调研沉淀。回答三个问题：**GT 长什么样？judge 怎么判？过程怎么量？**
> 每个项目给出对我们的映射——哪个判分通道借鉴了它。

## 0. 核心结论

**GT 是一条从紧到松的光谱：精确调用匹配 → 状态等价 → 终态断言集合 → 行为 rubric。**
业界主流已放弃"action 全匹配"（Anthropic 明确立场："grade what the agent produced, not the path it took"）；
正确姿势 = **终态断言为主 + 过程 rubric 为辅 + 负向规则兜底 + 参考解仅归因**。

---

## 1. 判分器与 rubric DSL（LLM judge 侧）

| 项目 | 核心设计 | 我们的映射 |
|---|---|---|
| **Prometheus 2**（arXiv 2405.01535） | 开源判分专用模型：用户自定义 score rubric + 1–5 直接评估/两两对比；与人类判断一致性最高的开源 judge | rubric 的结构化格式参考（criteria/description/scale）；支持本地 judge（私有集数据不出域） |
| **promptfoo llm-rubric** | 自然语言 rubric 断言 DSL；`agent-rubric` 变体允许判分器自带工具做 agentic 判分 | YAML 用例中 rubric 的自然语言写法；agentic judge 思路 |
| **DeepEval G-Eval** | CoT 链式思考判分（G-Eval 论文工业实现） | judge prompt 的 CoT 结构参考 |
| **Inspect AI scorers** | solver/scorer 分离 + 多种 scorer 可组合 | 判分通道的组合与注册机制（我们的 Scorer 协议） |

## 2. 基准侧 GT 形态（确定性断言侧）

| 项目 | GT 形态 | 我们的映射 |
|---|---|---|
| **BFCL v1/v2**（单轮） | expected 精确调用（AST 级 + possible answers 集合 + 类型等价）——选型期考核"单步工具调用能力"，无路径多样性 | 冻结回归用例（RC-xxxx）的参照形态 |
| **BFCL v3/v4**（多轮） | **状态等价**：执行模型调用后比对最终环境状态（多路径到达同一终态都算过）+ ROT（惩罚开关对消类无效操作）+ 死循环检测 | ROT ≈ 我们的 `useful_action_ratio`；死循环检测 ≈ 我们的熔断器；状态等价 ≈ 我们的终态断言通道 |
| **tau2-bench** | DB 写断言 + ACTION 参考解（仅对照，社区默认关闭——把参考解当唯一解是强假设） | 参考解只归因不判分的教训；DB 写断言思想 |
| **VitaBench**（ICLR 2026） | rubric 原子清单 + 滑窗状态向量（κ=0.828；无 rubric κ<0.07）；`expected_states.required_orders` 仅数据模型定义，**官方 evaluator 实际不做 DB 对比**（源码实证：final_state 传入后未消费），用 judge prompt 硬规则弥补（"订单类 rubric 必须确认真实下单成功"） | 滑窗判分器主体；官方 judge 三要素（任务指令/环境时间/订单确认硬规则）已吸收；终态断言是我们**超越官方**的增强通道 |
| **SWE-bench** | fail-to-pass + pass-to-pass 测试套件（任意改码路径，测试过即过） | coding 域接入时的确定性通道形态 |
| **WebArena / OSWorld** | 功能性终态检查（URL/元素/文件系统脚本） | 网页/OS 域的终态断言形态 |

## 3. 失效分类学（归因侧）

| 项目 | 内容 | 我们的映射 |
|---|---|---|
| **MAST**（arXiv 2503.13657，NeurIPS 2025） | 多 Agent 系统失效分类学：3 大类 14 种失效模式 + 大规模人工标注集；含"GT 难以核验"场景的标注方法论 | 归因分类学（§6.4）的多 Agent 扩展位（plan/subagent 事件分诊） |
| **VitaBench §6.3** | 单 Agent 工具任务失效分布：推理 61.8% / 工具 21.1% / 交互 7.9% / 噪声 9.2% | 归因分类学基底 |

## 4. 方法论文献

| 来源 | 核心主张 | 我们的映射 |
|---|---|---|
| **Anthropic: Demystifying evals for AI agents** | 评产出不评路径；grader 三型（code/model/human）；pass^k vs pass@k；20–50 个真实失败任务即可起步；CORE-Bench 因判分过死 42%→95% 的教训 | EDD 方法论、pass^k、freeze 回归（能力集毕业进回归集） |
| **《AI Engineering》EDD** | 先定义成功标准再构建 Agent（TDD 类比） | 框架第一性方法论与命名 |

## 5. 我们的判分架构（横评后的合成）

```
passed = 全部 rubric 满足（滑窗 judge，官方同款 + 三要素）
       AND 无轨迹规则违例（禁调/死循环/超轮次 —— 通道②，BFCL v3 ROT 同源）
       AND 无终态断言违例（expected_states 字段级比对 —— 超越 VitaBench 官方的增强通道）
```

| 设计决策 | 来源 |
|---|---|
| 滑窗 + 粘滞状态向量 + 全有或全无 | VitaBench（κ=0.828 实证） |
| judge prompt 含任务指令/环境时间/订单确认硬规则 | VitaBench 官方 evaluator 源码 |
| 终态断言通道（官方未做，我们补上） | BFCL v3 状态等价 + tau2 DB 写断言思想 |
| 有效动作比/循环熔断 | BFCL v3 ROT + 我们的 openai-loop 实战（297 连击 bad case） |
| 错误三分类 tool/agent/infra | 自研（infra 豁免是生产视角） |
| 参考解仅归因 | tau2 ACTION 通道教训 |
| 部分 score = 满足数/总数 | VitaBench rubric 级明细 + Anthropic 部分得分主张 |
