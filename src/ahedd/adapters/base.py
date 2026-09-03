"""被测 Agent 适配层基类（车道一：进程内）。

契约要点：适配器只负责"驱动被测 Agent 消费工具完成任务"。
轨迹采集由 runner 通过 TrajectoryRecorder.wrap_tool 包装工具注入，
适配器只需在合适时机调用 recorder.note() 补记对话消息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# 兼容再导出：ToolDefinition 的概念归属是环境层（ahedd.env.tools），
# 早期版本定义于此，保留导入路径以免破坏外部引用。
from ahedd.env.tools import JsonSchema, ToolDefinition  # noqa: F401


@dataclass
class TaskInput:
    """交给被测 Agent 的任务输入。"""

    task_id: str
    instruction: str
    system_prompt: str | None = None


@dataclass
class AgentResult:
    """被测 Agent 的终局输出。"""

    final_message: str
    stop_reason: str = "stop"  # stop | max_steps | error
    usage_total: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class AgentAdapter(Protocol):
    """所有进程内被测对象的基类（结构化协议，无需显式继承）。

    双工具命名空间：env tools 由框架注入并录制；被测 Harness 自带的内部模块
    （planning / subagent / memory 等）不注入、不拦截，其调用由适配器经
    recorder.note() 以专属 kind 入轨（见 trace.schema.CANONICAL_STEP_KINDS）。
    """

    name: str

    async def run(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: Any | None = None,
    ) -> AgentResult:
        """执行任务：消费工具直至完成，返回终局输出。"""
        ...
