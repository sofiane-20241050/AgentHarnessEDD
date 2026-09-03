"""模型访问层基类（接缝预留）。

当前唯一实现是 OpenAIClient（OpenAI SDK 兼容端点，覆盖 vLLM / OpenRouter /
各家官方 API）。未来要接其他 SDK（Anthropic 原生 / Gemini 原生 / 本地推理）
时，实现本协议 + 在 make_client 工厂注册即可，上层（适配器 / 用户模拟器 /
判分器）代码零改动。

消息与工具的线上格式约定：**OpenAI chat 格式**（role/content/tool_calls/
tool_call_id）。非 OpenAI 实现负责在协议内部做格式翻译。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ahedd.trace.schema import Usage


@dataclass
class ToolCall:
    """模型发出的一次工具调用意图。"""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """一次 chat 补全的归一化返回。

    reasoning_content：思考型模型（vLLM 部署的 Qwen3/DeepSeek-R1 等）的原始思考链，
    与最终回答分离返回；非思考模型为 None。
    """

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # 原始响应（调试 / 适配器特有信息用）


@runtime_checkable
class LLMClient(Protocol):
    """模型客户端契约：chat 一条路走到底。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        ...
