"""模型访问层入口：客户端工厂（provider 分发的唯一位置）+ 各生态模型桥接。

要新增一个 SDK：实现 llm.base.LLMClient，然后在这里的 _FACTORIES 注册。
要给 LangChain/tau 系 Harness 构造原生模型对象：见 llm.bridges。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ahedd.llm.base import LLMClient, LLMResponse, ToolCall
from ahedd.llm.bridges import build_langchain_model, build_tau_provider
from ahedd.llm.fake import FakeLLMClient
from ahedd.llm.openai_client import OpenAIClient

if TYPE_CHECKING:
    from ahedd.config import ModelSpec


def make_client(spec: ModelSpec) -> LLMClient:
    factories: dict[str, type] = {
        "openai": OpenAIClient,  # OpenAI SDK 兼容端点（vLLM / OpenRouter / 官方 API）
        "fake": FakeLLMClient,   # 仅供测试：脚本回放
    }
    try:
        cls = factories[spec.provider]
    except KeyError:
        raise ValueError(
            f"unknown provider: {spec.provider!r}, available: {sorted(factories)}; "
            "新 SDK 请实现 LLMClient 协议并在 make_client 注册（预留接缝）"
        ) from None
    if cls is FakeLLMClient:
        return FakeLLMClient()  # type: ignore[return-value]
    return cls(spec)  # type: ignore[call-arg]


__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "ToolCall",
    "build_langchain_model",
    "build_tau_provider",
    "make_client",
]
