"""OpenAI SDK 兼容客户端——当前唯一的 LLMClient 实现。

AsyncOpenAI + base_url 直连任意兼容端点（vLLM / OpenRouter / 官方 API）。
api key 从 ModelSpec.api_key_env 指定的环境变量读取，不落明文。
"""

from __future__ import annotations

import json
from typing import Any

from ahedd.config import ModelSpec
from ahedd.llm.base import LLMResponse, ToolCall
from ahedd.trace.schema import Usage


class OpenAIClient:
    def __init__(self, spec: ModelSpec) -> None:
        from openai import AsyncOpenAI

        self.spec = spec
        self._client = AsyncOpenAI(
            base_url=spec.base_url,
            api_key=spec.resolve_api_key(),
            default_headers=spec.extra_headers or None,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": self.spec.temperature,
        }
        if tools:
            params["tools"] = tools
        if self.spec.max_tokens:
            params["max_tokens"] = self.spec.max_tokens
        params.update(kwargs)

        resp = await self._client.chat.completions.create(**params)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                args=_parse_arguments(tc.function.arguments),
            )
            for tc in (msg.tool_calls or [])
        ]
        usage = Usage()
        if resp.usage:
            usage.input_tokens = resp.usage.prompt_tokens or 0
            usage.output_tokens = resp.usage.completion_tokens or 0
        return LLMResponse(
            content=msg.content,
            reasoning_content=_extract_reasoning(msg),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            raw=resp,
        )


def _extract_reasoning(msg: Any) -> str | None:
    """思考链字段名在不同后端不统一，按序尝试：
    message.reasoning_content（DeepSeek 惯例 / 标准 OpenAI 属性位）
    -> extra["reasoning_content"] -> extra["reasoning"]（部分 vLLM 版本）。取不到返回 None。
    """
    candidates: list[Any] = [
        getattr(msg, "reasoning_content", None),
    ]
    extra = getattr(msg, "model_extra", None) or {}
    candidates.append(extra.get("reasoning_content"))
    candidates.append(extra.get("reasoning"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_malformed_arguments": raw}
