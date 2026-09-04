"""环境层基类：工具集 + 虚拟数据库的确定性内核。

环境是评测中唯一"说真话"的部分：工具是确定性 Python 函数，
数据库可快照、可 diff——这是确定性断言通道的基础。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ahedd.env.tools import ToolDefinition


@runtime_checkable
class Environment(Protocol):
    """仿真环境契约。实现方保证：相同 seed 下 reset 结果可复现。"""

    domain: str

    def tools(self) -> list[ToolDefinition]:
        """暴露给被测 Agent 的全部工具（OpenAI function-calling 形态）。"""
        ...

    async def reset(self, seed: int | None = None) -> None:
        """重置到任务初始状态（数据库种子 / 系统时间等）。"""
        ...

    def snapshot(self) -> dict[str, Any]:
        """全量状态快照，用于执行前后对比。"""
        ...

    def diff(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """两个快照的差异（写操作断言 / 报告 State Diff 视图的数据源）。"""
        ...


def default_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """朴素的一层 diff：新增 / 删除 / 修改的键。环境实现方可覆写更细粒度版本。"""
    changed: dict[str, Any] = {}
    for key in set(before) | set(after):
        if key not in before:
            changed[key] = {"op": "added", "after": after[key]}
        elif key not in after:
            changed[key] = {"op": "removed", "before": before[key]}
        elif before[key] != after[key]:
            changed[key] = {"op": "modified", "before": before[key], "after": after[key]}
    return changed
