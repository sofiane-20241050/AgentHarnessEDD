"""车道一默认适配器：通用 OpenAI 兼容 function-calling 循环。

标准工具调用循环：LLM -> tool_calls -> 执行（经录制包装）-> 回填 -> ... -> 终答。
依赖注入 LLMClient（OpenAI 实现 / 测试用 Fake），本适配器不感知具体 SDK。
"""

from __future__ import annotations

import json
from typing import Any

from ahedd.adapters import register_adapter
from ahedd.adapters.base import AgentResult, TaskInput, ToolDefinition
from ahedd.env.tools import ToolRegistry
from ahedd.llm.base import LLMClient
from ahedd.trace.schema import TrajectoryRecorder, Usage

DEFAULT_SYSTEM_PROMPT = (
    "你是一个业务助手。使用提供的工具完成用户的请求；"
    "无法或不应执行的操作要明确告知用户并说明原因。"
)


class OpenAILoopAdapter:
    name = "openai-loop"

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 300,
        chat_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        # 透传给 client.chat 的额外参数，如 max_tokens / extra_body
        # （vLLM 思考模型可用 extra_body={"chat_template_kwargs": {"enable_thinking": False}}）
        self.chat_kwargs = chat_kwargs or {}

    async def run(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        messages: list[dict[str, Any]] = []
        if task.system_prompt or self.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt or self.system_prompt})
        messages.append({"role": "user", "content": task.instruction})

        registry = ToolRegistry(tools)
        total = Usage()

        for _turn in range(self.max_turns):
            try:
                resp = await self.client.chat(messages, registry.to_openai_tools() or None, **self.chat_kwargs)
            except Exception as exc:
                # 网络/限流/超时等基础设施错误：分类入轨后上抛（run 由 runner 收卷）
                from ahedd.trace.errors import classify_exception

                if recorder:
                    recorder.note(
                        "error",
                        content=f"{type(exc).__name__}: {exc}",
                        error_kind=classify_exception(exc),
                        stop_reason="error",
                    )
                raise
            total = Usage(
                input_tokens=total.input_tokens + resp.usage.input_tokens,
                output_tokens=total.output_tokens + resp.usage.output_tokens,
                cost_usd=total.cost_usd + resp.usage.cost_usd,
            )
            messages.append(_assistant_message(resp))

            if not resp.tool_calls:
                if recorder:
                    recorder.note(
                        "assistant",
                        content=resp.content or "",
                        reasoning=resp.reasoning_content,
                        stop_reason=resp.finish_reason,
                        usage=resp.usage,
                    )
                return AgentResult(
                    final_message=resp.content or "",
                    stop_reason="stop" if resp.finish_reason == "stop" else resp.finish_reason,
                    usage_total={
                        "input_tokens": total.input_tokens,
                        "output_tokens": total.output_tokens,
                        "cost_usd": total.cost_usd,
                    },
                )

            # 每次补全记一条 assistant 事件（可能无文本，携带思考链与本轮用量）
            if recorder:
                recorder.note(
                    "assistant", content=resp.content or "", reasoning=resp.reasoning_content, usage=resp.usage
                )
            for tc in resp.tool_calls:
                messages.append(await self._execute(tc, registry, recorder))

        if recorder:
            recorder.note("error", content="max_turns exceeded", stop_reason="max_steps", error_kind="agent")
        return AgentResult(
            final_message="",
            stop_reason="max_steps",
            usage_total={
                "input_tokens": total.input_tokens,
                "output_tokens": total.output_tokens,
                "cost_usd": total.cost_usd,
            },
        )

    async def _execute(
        self,
        tc: Any,
        registry: ToolRegistry,
        recorder: TrajectoryRecorder | None,
    ) -> dict[str, Any]:
        tool = registry.get(tc.name)
        if tool is None:
            if recorder:
                recorder.note("error", content=f"unknown tool: {tc.name}", tool_name=tc.name, error_kind="agent")
            return _tool_message(tc.id, {"ok": False, "error": f"unknown tool: {tc.name}"})
        try:
            result = await tool.func(**tc.args)
            return _tool_message(tc.id, result)
        except TypeError as exc:  # 参数签名不匹配（schema/调用不一致）
            if recorder:
                recorder.note("error", content=f"TypeError: {exc}", tool_name=tc.name, error_kind="agent")
            return _tool_message(tc.id, {"ok": False, "error": f"bad arguments: {exc}"})


def _assistant_message(resp: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)},
            }
            for tc in resp.tool_calls
        ]
    return msg


def _tool_message(call_id: str, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False, default=str),
    }


@register_adapter("openai-loop")
def _factory() -> OpenAILoopAdapter:
    raise TypeError("openai-loop 需要模型客户端：OpenAILoopAdapter(make_client(spec))")
