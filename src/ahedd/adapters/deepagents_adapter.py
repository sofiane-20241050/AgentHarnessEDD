"""DeepAgents（langchain-ai/deepagents，LangGraph 内核）进程内适配器。

集成方式（对齐仓库 libs/deepagents 实测 API）：
  - create_deep_agent(model=ChatOpenAI(...), tools=[StructuredTool...], system_prompt=...)
    返回 CompiledStateGraph，await agent.ainvoke({"messages": [...]}) 驱动
  - 内置工具由必需中间件提供（FilesystemMiddleware -> ls/read_file/write_file/edit_file/
    glob/grep/delete/execute；SubAgentMiddleware -> task），tools= 参数只增不减；
    我们传入的 system_prompt 是 USER 段，DeepAgents 会追加 profile 的 BASE/SUFFIX 段（补充而非覆盖）
  - 轨迹后置重建：ainvoke 结束后按 result["messages"] 重建完整步骤序列，
    每条 AIMessage 携带自身 usage_metadata（token 逐步可见）；
    内置工具调用按 _HARNESS_TOOL_KINDS 映射为 plan/subagent/memory 事件
    （双工具命名空间，见调研报告 §3.6）

自定义扩展（透传给 create_deep_agent）：middleware / subagents / create_kwargs。

安装：pip install "agentharness-edd[deepagents]"
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ahedd.adapters import register_adapter
from ahedd.adapters.base import AgentResult, TaskInput
from ahedd.adapters.openai_loop import DEFAULT_SYSTEM_PROMPT
from ahedd.env.tools import ToolDefinition
from ahedd.trace.schema import StepRecord, Usage

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ahedd.config import ModelSpec
    from ahedd.trace.schema import TrajectoryRecorder

_INSTALL_HINT = 'pip install "agentharness-edd[deepagents]"'

# harness 原生工具 -> 轨迹 type 的映射（本项目的归因分类学，见调研报告 §6.4；
# 非任何外部标准，实例可用 harness_tool_kinds 参数覆盖/扩展）
DEFAULT_HARNESS_TOOL_KINDS: dict[str, str] = {
    "write_todos": "plan",
    "todo_write": "plan",
    "task": "subagent",
    "ls": "memory",
    "read_file": "memory",
    "write_file": "memory",
    "edit_file": "memory",
    "glob": "memory",
    "grep": "memory",
}

_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class DeepAgentsAdapter:
    name = "deepagents"

    def __init__(
        self,
        model_spec: ModelSpec | None = None,
        *,
        model: Any = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 300,
        disable_thinking: bool = False,
        middleware: Any = None,
        subagents: Any = None,
        create_kwargs: dict[str, Any] | None = None,
        harness_tool_kinds: dict[str, str] | None = None,
    ) -> None:
        """模型与 Harness 组装解耦：
        :param model_spec: 端点配置（经 llm.build_langchain_model 桥接为 ChatOpenAI）
        :param model: 直接传入自建的原生模型对象（BaseChatModel，绕过 ModelSpec），二选一
        :param middleware: 追加/替换 LangChain AgentMiddleware（None 用 DeepAgents 默认栈）
        :param subagents: 自定义 SubAgent 列表
        :param create_kwargs: 其余 create_deep_agent 参数透传（backend/skills/memory/...）
        :param harness_tool_kinds: 内置工具 -> 轨迹 type 的映射覆盖
        """
        if model is None and model_spec is None:
            raise ValueError("model_spec 与 model 至少提供一个")
        self.model_spec = model_spec
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.disable_thinking = disable_thinking
        self.middleware = middleware
        self.subagents = subagents
        self.create_kwargs = create_kwargs or {}
        self.harness_tool_kinds = dict(DEFAULT_HARNESS_TOOL_KINDS)
        if harness_tool_kinds:
            self.harness_tool_kinds.update(harness_tool_kinds)

    # ---- 组装 ----

    def _to_lc_tool(self, tool: ToolDefinition) -> Any:
        """ToolDefinition -> LangChain 原生 StructuredTool（@tool 装饰器产物的同类对象）。"""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            coroutine=tool.func,
            name=tool.name,
            description=tool.description,
            args_schema=_schema_to_args_model(tool.name, tool.parameters),
        )

    # ---- 执行 ----

    async def run(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        try:
            from deepagents import create_deep_agent
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        graph_errors: tuple[type[BaseException], ...] = ()
        try:
            from langgraph.errors import GraphRecursionError

            graph_errors = (GraphRecursionError,)
        except ImportError:
            pass

        env_tool_names = {t.name for t in tools}
        extra: dict[str, Any] = {}
        if self.middleware is not None:
            extra["middleware"] = self.middleware
        if self.subagents is not None:
            extra["subagents"] = self.subagents
        extra.update(self.create_kwargs)
        from ahedd.llm import build_langchain_model

        model = self.model if self.model is not None else build_langchain_model(
            self.model_spec, disable_thinking=self.disable_thinking  # type: ignore[arg-type]
        )
        agent = create_deep_agent(
            model=model,
            tools=[self._to_lc_tool(t) for t in tools],
            system_prompt=task.system_prompt or self.system_prompt,
            **extra,
        )

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task.instruction}]},
                config={"recursion_limit": self.max_turns * 2},
            )
        except graph_errors:  # type: ignore[misc]
            if recorder:
                recorder.note(
                    "error",
                    content="recursion limit reached (max_turns)",
                    error_kind="agent",
                )
            return AgentResult(final_message="", stop_reason="max_steps")

        steps, final_text, usage_total = _rebuild_steps(result["messages"], env_tool_names, self.harness_tool_kinds)
        if recorder:
            recorder.trajectory.steps = steps  # 用重建序列替换执行期的零散记录，保证顺序与用量完整
        return AgentResult(final_message=final_text, stop_reason="stop", usage_total=usage_total)


def _rebuild_steps(
    messages: Any,
    env_tool_names: set[str],
    kind_map: dict[str, str],
) -> tuple[list[StepRecord], str, dict[str, float]]:
    """从 LangGraph 结果消息重建完整轨迹：每条 AIMessage 一条 assistant 事件（含自身用量），
    env 工具与 harness 原生工具的调用/结果按真实顺序入轨。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    steps: list[StepRecord] = []
    call_id_to_name: dict[str, str] = {}
    final_text = ""
    total_in = total_out = 0

    def add(type_: str, **fields: Any) -> None:
        steps.append(StepRecord(index=len(steps), type=type_, **fields))  # type: ignore[call-arg]

    for msg in messages:
        if isinstance(msg, HumanMessage):
            add("user", content=_content_text(msg.content))
        elif isinstance(msg, AIMessage):
            meta = msg.usage_metadata or {}
            usage = Usage(
                input_tokens=int(meta.get("input_tokens", 0) or 0),
                output_tokens=int(meta.get("output_tokens", 0) or 0),
            )
            total_in += usage.input_tokens
            total_out += usage.output_tokens
            text = _content_text(msg.content)
            reasoning = (msg.additional_kwargs or {}).get("reasoning_content")  # vLLM 思考链
            add(
                "assistant",
                content=text,
                reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
                usage=usage,
            )
            if text.strip():
                final_text = text
            for tc in msg.tool_calls or []:
                tool_name = tc.get("name") or ""
                call_id_to_name[tc.get("id") or ""] = tool_name
                step_type = "tool_call" if tool_name in env_tool_names else kind_map.get(tool_name, "tool_call")
                add(step_type, tool_name=tool_name, tool_args=tc.get("args"))
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None) or call_id_to_name.get(msg.tool_call_id, "")
            add("tool_result", tool_name=tool_name, tool_result=_parse_content(msg.content))

    return steps, final_text, {"input_tokens": total_in, "output_tokens": total_out, "cost_usd": 0.0}


def _parse_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content or "")


def _schema_to_args_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """JSON Schema -> pydantic args 模型（StructuredTool 需要）。"""
    from pydantic import Field, create_model

    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, field_def in properties.items():
        py_type = _JSON_TYPES.get(field_def.get("type", "string"), str)
        if field_name in required:
            fields[field_name] = (py_type, Field(..., description=field_def.get("description", "")))
        else:
            fields[field_name] = (py_type | None, Field(None, description=field_def.get("description", "")))
    return create_model(f"{tool_name}_args", **fields)


@register_adapter("deepagents")
def _factory() -> DeepAgentsAdapter:
    raise TypeError("deepagents 需要模型端点参数：请直接构造 DeepAgentsAdapter(model_spec)")
