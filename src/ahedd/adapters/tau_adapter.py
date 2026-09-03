"""tau（huggingface/tau，Pi 的 Python 移植版）进程内适配器。

集成方式（对齐仓库 src/tau_agent 实测 API）："事件流即契约"：
  - AgentHarness(AgentHarnessConfig(provider, model, system, tools))，
    async for event in harness.prompt(...) 消费类型化事件
  - 环境工具包装为 AgentTool（execute_fn 内调用 runner 已录制的函数）
  - 事件流中映射 assistant 消息入轨；env 工具调用由录制包装记录（单一事实来源）
  - tau_ai.OpenAICompatibleProvider 直连 OpenAI 兼容端点（base_url 含 /v1）

安装：pip install "agentharness-edd[tau]"（需 Python >= 3.12）
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ahedd.adapters import register_adapter
from ahedd.adapters.base import AgentResult, TaskInput
from ahedd.adapters.openai_loop import DEFAULT_SYSTEM_PROMPT
from ahedd.env.tools import ToolDefinition
from ahedd.trace.schema import Usage

if TYPE_CHECKING:
    from ahedd.config import ModelSpec
    from ahedd.trace.schema import TrajectoryRecorder

_INSTALL_HINT = 'pip install "agentharness-edd[tau]"'


class TauAdapter:
    name = "tau"

    def __init__(
        self,
        model_spec: ModelSpec | None = None,
        *,
        provider: Any = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 50,
        config_extras: dict[str, Any] | None = None,
    ) -> None:
        """模型与 Harness 组装解耦：
        :param model_spec: 端点配置（经 llm.build_tau_provider 桥接）
        :param provider: 直接传入自建的 tau ModelProvider（绕过 ModelSpec），二选一
        :param config_extras: 透传给 AgentHarnessConfig 的额外字段
            （before_tool_call / after_tool_call / queue_mode / session_id 等，见 tau_agent.harness）
        """
        if provider is None and model_spec is None:
            raise ValueError("model_spec 与 provider 至少提供一个")
        self.model_spec = model_spec
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.config_extras = config_extras or {}

    def _to_tau_tool(self, tool: ToolDefinition) -> Any:
        from tau_agent.tools import AgentTool, AgentToolResult

        async def _execute(
            tool_call_id: str, arguments: Any, signal: Any = None, on_update: Any = None
        ) -> Any:
            try:
                result = await tool.func(**dict(arguments))
                text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                return AgentToolResult(content=text, details=result)
            except Exception as exc:  # noqa: BLE001 - 工具错误应回流给模型而非崩溃
                return AgentToolResult(content=f"error: {exc}", details={"ok": False, "error": str(exc)})

        return AgentTool(
            name=tool.name,
            label=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            execute_fn=_execute,
        )

    async def run(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        try:
            from tau_agent import AgentHarness, AgentHarnessConfig
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        from ahedd.llm import build_tau_provider

        owns_provider = self.provider is None
        provider = self.provider if self.provider is not None else build_tau_provider(self.model_spec)  # type: ignore[arg-type]
        config_kwargs: dict[str, Any] = {
            "provider": provider,
            "model": self.model_spec.model if self.model_spec else getattr(provider, "model", "custom"),
            "system": task.system_prompt or self.system_prompt or "You are a helpful assistant.",
            "tools": [self._to_tau_tool(t) for t in tools],
            "max_turns": self.max_turns,
        }
        config_kwargs.update(self.config_extras)  # 用户透传（before_tool_call 等）
        harness = AgentHarness(AgentHarnessConfig(**config_kwargs))

        final_text = ""
        usage_total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        try:
            async for event in harness.prompt(task.instruction):
                if event.type == "message_end":
                    message = event.message
                    if not type(message).__name__.lower().startswith("assistant"):
                        continue
                    text = _message_text(message)
                    thinking = getattr(message, "thinking_text", None)  # tau ThinkingContent 块拼接
                    usage = _tau_usage(message)
                    usage_total["input_tokens"] += usage.input_tokens
                    usage_total["output_tokens"] += usage.output_tokens
                    usage_total["cost_usd"] += usage.cost_usd
                    if text or thinking:
                        if recorder:
                            recorder.note(
                                "assistant",
                                content=text,
                                reasoning=thinking if isinstance(thinking, str) and thinking else None,
                                usage=usage,
                            )
                        final_text = text or final_text
        finally:
            if owns_provider:
                await provider.aclose()  # 自建的 provider 才由我们关闭，用户自带的不动
        return AgentResult(final_message=final_text, stop_reason="stop", usage_total=usage_total)


def _tau_usage(message: Any) -> Usage:
    """tau AssistantMessage.usage(input/output/cost.total) -> 统一 Usage。"""
    raw = getattr(message, "usage", None)
    if raw is None:
        return Usage()
    cost = getattr(raw, "cost", None)
    return Usage(
        input_tokens=int(getattr(raw, "input", 0) or 0),
        output_tokens=int(getattr(raw, "output", 0) or 0),
        cost_usd=float(getattr(cost, "total", 0.0) or 0.0),
    )


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    parts.append(block_text)
        return "".join(parts)
    return ""


@register_adapter("tau")
def _factory() -> TauAdapter:
    raise TypeError("tau 需要模型端点参数：请直接构造 TauAdapter(model_spec)")
