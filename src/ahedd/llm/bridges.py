"""ModelSpec -> 各 Harness 生态原生模型对象的桥接（模型访问层的延伸）。

适配器只负责 Harness 组装与轨迹映射；"用哪个模型、怎么构造"统一在这里：
  - build_langchain_model(spec) -> ChatOpenAI        （deepagents 等 LangChain 系）
  - build_tau_provider(spec)     -> OpenAICompatibleProvider（tau 系）

调用方也可以完全绕开 ModelSpec，直接给适配器传自建的原生模型对象。
重度依赖均在函数内延迟导入，未安装对应 extra 不影响核心包。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ahedd.config import ModelSpec


def build_langchain_model(spec: ModelSpec, *, disable_thinking: bool = False) -> Any:
    """ModelSpec -> langchain_openai.ChatOpenAI（BaseChatModel）。"""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {}
    if disable_thinking:
        # vLLM Qwen3 系：关闭思考
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(
        model=spec.model,
        base_url=spec.base_url,
        api_key=spec.resolve_api_key(),
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        **kwargs,
    )


def build_tau_provider(spec: ModelSpec) -> Any:
    """ModelSpec -> tau_ai.OpenAICompatibleProvider（base_url 需含 /v1）。"""
    from tau_ai import OpenAICompatibleProvider
    from tau_ai.env import OpenAICompatibleConfig

    return OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key=spec.resolve_api_key(),
            base_url=spec.base_url or "https://api.openai.com/v1",
            max_tokens=spec.max_tokens,
        )
    )
