"""脚本化假客户端：测试与离线演示用，零 API 成本、完全确定性。

每次 chat 按顺序弹出一个预置响应；收到的 messages 会被记录，供断言。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ahedd.llm.base import LLMResponse, ToolCall
from ahedd.trace.schema import Usage


class FakeLLMClient:
    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        """script 每项形如::

            {"content": "...", "finish_reason": "stop"}
            {"tool_calls": [{"id": "c1", "name": "get_order", "args": {...}}]}
        """
        self._script: deque[dict[str, Any]] = deque(script or [])
        self.received: list[list[dict[str, Any]]] = []
        self.received_tools: list[list[dict[str, Any]] | None] = []

    def enqueue(self, item: dict[str, Any]) -> None:
        self._script.append(item)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.received.append([dict(m) for m in messages])
        self.received_tools.append([dict(t) for t in tools] if tools else None)
        if not self._script:
            raise AssertionError("FakeLLMClient script exhausted")
        item = self._script.popleft()
        return LLMResponse(
            content=item.get("content"),
            reasoning_content=item.get("reasoning_content"),
            tool_calls=[ToolCall(**tc) for tc in item.get("tool_calls", [])],
            finish_reason=item.get("finish_reason", "tool_calls" if item.get("tool_calls") else "stop"),
            usage=Usage(input_tokens=item.get("input_tokens", 1), output_tokens=item.get("output_tokens", 1)),
        )
