"""LLM 用户模拟器：为"双 LLM 交互"数据集（vita / tau2 系）扮演用户。

设计约定（对齐 tau2-bench / VitaBench 论文的用户模拟器形态）：
  - 剧本（instruction，第二人称场景设定）与画像（persona/traits）是模拟器的"底牌"，
    只对模拟器可见，不直接暴露给被测 Agent
  - 模拟器说"人话"：每次只说用户自然会说的下一句（简短、符合人设），开局只透露
    用户会主动说的部分——隐含约束需要 Agent 追问才给出（这正是"澄清"考察点的来源）
  - 被问到剧本外信息时回答不知道；任务完成或对话自然结束时输出 ###STOP###
  - 模型端点复用 models 配置的 user_simulator 角色（OpenAI 兼容）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ahedd.llm.base import LLMClient
from ahedd.trace.schema import Usage

if TYPE_CHECKING:
    from ahedd.datasets.base import TaskCase

STOP_TOKEN = "###STOP###"

_SYSTEM_PROMPT = """你在扮演一个真实用户，正在与一个业务客服 Agent 对话。

# 你的任务设定（只属于你，不要直接复述给 Agent）
{scenario}

# 行为规则
1. 每次回复只说用户自然会说的下一句话，简短口语化，符合人设（{persona}）。
2. 开场只主动说出用户会主动说的需求；设定中的隐含约束（时间/忌口/偏好等）
   只有当 Agent 问到或确有必要时才透露——用户不会一次性交代所有信息。
3. Agent 问到你设定里没有的信息，就说不知道（不要编造）。
4. Agent 完成了你的需求，或你确认没有更多要求时，回复：{stop}
5. 不要输出任何解释、括号说明或扮演标记。"""


class UserSimulator:
    """逐轮扮演用户：start() 产出开场白，reply() 回应 Agent 的每条消息。"""

    def __init__(
        self,
        client: LLMClient,
        case: TaskCase,
        *,
        max_dialog_turns: int = 30,
        chat_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.case = case
        self.max_dialog_turns = max_dialog_turns
        self.chat_kwargs = chat_kwargs or {}
        self.history: list[dict[str, str]] = []  # 模拟器侧的对话记忆（user/agent 交替）
        self.total_usage = Usage()
        self.turns = 0

    async def _chat(self) -> str:
        from ahedd.llm.base import LLMResponse  # noqa: F401  (类型文档用)

        system = _SYSTEM_PROMPT.format(
            scenario=_scenario_text(self.case),
            persona=_persona_text(self.case) or "普通用户",
            stop=STOP_TOKEN,
        )
        messages = [{"role": "system", "content": system}, *self.history]
        resp = await self.client.chat(messages, **self.chat_kwargs)
        self.total_usage = Usage(
            input_tokens=self.total_usage.input_tokens + resp.usage.input_tokens,
            output_tokens=self.total_usage.output_tokens + resp.usage.output_tokens,
            cost_usd=self.total_usage.cost_usd + resp.usage.cost_usd,
        )
        self.turns += 1
        return (resp.content or "").strip()

    async def start(self) -> str:
        """产出第一句用户话语（模拟器内部先自问"用户会怎么开口"）。"""
        self.history.append(
            {"role": "user", "content": "（对话开始。请输出你作为用户的第一句话。）"}
        )
        opening = await self._chat()
        self.history.append({"role": "assistant", "content": opening})
        return opening

    async def reply(self, agent_message: str) -> str:
        """回应 Agent 的一条消息；返回下一句用户话语（可能含 ###STOP###）。"""
        self.history.append({"role": "user", "content": f"客服Agent说：{agent_message}"})
        answer = await self._chat()
        self.history.append({"role": "assistant", "content": answer})
        return answer

    @property
    def finished(self) -> bool:
        return any(STOP_TOKEN in m["content"] for m in self.history if m["role"] == "assistant") or (
            self.turns >= self.max_dialog_turns
        )


def _scenario_text(case: TaskCase) -> str:
    parts = [f"你的原始任务设定：{case.instruction}"]
    traits = case.user_scenario.traits if case.user_scenario else {}
    if traits:
        detail = "；".join(f"{k}: {v}" for k, v in list(traits.items())[:12])
        parts.append(f"你的个人资料：{detail}")
    return "\n".join(parts)


def _persona_text(case: TaskCase) -> str:
    return (case.user_scenario.persona if case.user_scenario else "") or ""


def is_stop(message: str) -> bool:
    return STOP_TOKEN in message
