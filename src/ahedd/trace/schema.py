"""统一轨迹 Schema：评测的"先存轨迹、后判分"基石。

所有车道（进程内 / MCP / RPC）产出的轨迹都归一到本模块的事件模型，
判分器、报告、回归冻结消费同一份数据。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ahedd.env.tools import ToolDefinition

# canonical 事件类别。不设封闭枚举：现代 Harness 的内部模块事件
# （plan / subagent / memory / handoff 等）允许作为自定义 type 入轨（见调研报告 §3.6）。
CANONICAL_STEP_TYPES = (
    "user", "assistant", "tool_call", "tool_result", "plan", "subagent", "memory", "error",
)
StepType = str


class Usage(BaseModel):
    """单步 token/成本计量。非进程内车道无法精确计量时保持默认 0。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class StepRecord(BaseModel):
    """轨迹中的一个原子事件。

    reasoning 为思考型模型的原始思考链（无则不落盘）；
    error_kind 仅 error 事件携带：tool / agent / infra（见 trace.errors）。
    """

    index: int
    type: StepType
    content: str = ""
    reasoning: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    stop_reason: str | None = None
    error_kind: str | None = None
    usage: Usage = Field(default_factory=Usage)
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunMeta(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str
    domain: str = ""
    dataset: str = ""
    adapter: str = ""
    agent_model: str = ""
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_usage: Usage = Field(default_factory=Usage)


class Trajectory(BaseModel):
    """一次任务的完整轨迹。JSONL 落盘格式：首行 RunMeta，其后每行一个 StepRecord。"""

    meta: RunMeta
    steps: list[StepRecord] = Field(default_factory=list)


class TrajectoryRecorder:
    """轨迹录制器。

    用法：runner 层通过 :meth:`wrap_tool` 包装环境工具后交给适配器，
    工具调用/返回即被自动记录；对话消息由适配器调用 :meth:`note` 补记。
    """

    def __init__(self, meta: RunMeta) -> None:
        self.trajectory = Trajectory(meta=meta)

    def note(self, type: str, content: str = "", **fields: Any) -> StepRecord:
        record = StepRecord(index=len(self.trajectory.steps), type=type, content=content, **fields)
        self.trajectory.steps.append(record)
        return record

    def wrap_tool(self, tool: ToolDefinition) -> ToolDefinition:
        """返回带录制能力的工具包装。"""
        inner = tool.func

        async def recorded(**kwargs: Any) -> Any:
            from ahedd.trace.errors import classify_exception

            self.note("tool_call", tool_name=tool.name, tool_args=kwargs)
            try:
                result = await inner(**kwargs)
            except Exception as exc:
                # MCP/网络工具的异常可能是 infra，参数不匹配是 agent，其余归 tool
                self.note(
                    "error",
                    content=f"{type(exc).__name__}: {exc}",
                    tool_name=tool.name,
                    error_kind=classify_exception(exc),
                )
                raise
            self.note("tool_result", tool_name=tool.name, tool_result=result)
            return result

        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            func=recorded,
        )

    def dump_jsonl(self, path: str) -> None:
        """落盘格式：首行 RunMeta，其后每行一个 StepRecord。"""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(self.trajectory.meta.model_dump_json() + "\n")
            for step in self.trajectory.steps:
                f.write(step.model_dump_json(exclude_none=True) + "\n")


def load_jsonl_trajectory(path: str) -> Trajectory:
    """读回落盘轨迹（离线重判 / 报告 / 冻结回归用例用）。"""
    import json
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty trajectory file: {path}")
    meta = RunMeta.model_validate(json.loads(lines[0]))
    steps = [StepRecord.model_validate(json.loads(ln)) for ln in lines[1:] if ln.strip()]
    return Trajectory(meta=meta, steps=steps)


__all__ = [
    "CANONICAL_STEP_TYPES",
    "RunMeta",
    "StepRecord",
    "StepType",
    "Trajectory",
    "TrajectoryRecorder",
    "Usage",
    "load_jsonl_trajectory",
]
