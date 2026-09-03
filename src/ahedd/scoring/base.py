"""判分双通道基类。

通道一（主）：rubric 滑动窗口 LLM 判分 —— 覆盖推荐/规划等不改变数据库
但至关重要的行为目标（终态哈希一刀切会漏判，见调研报告 §2.9）。
通道二（辅）：确定性断言 —— 写操作校验 + 轨迹规则，零成本零幻觉。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ahedd.datasets.base import TaskCase
from ahedd.trace.schema import Trajectory


class RubricResult(BaseModel):
    """单条 rubric 的判定。"""

    key: str
    description: str
    satisfied: bool
    evidence_turn: int | None = None  # 满足/违反发生在第几轮（诊断定位用）


class ScoreReport(BaseModel):
    """判分输出：任务级结论 + 双通道明细。"""

    task_id: str
    passed: bool = False
    score: float = 0.0                        # 全有或全无：全部 rubric 满足 = 1.0
    rubric_results: list[RubricResult] = Field(default_factory=list)
    rule_violations: list[str] = Field(default_factory=list)  # 确定性通道违例
    judge_meta: dict[str, str] = Field(default_factory=dict)  # 判分模型/窗口参数等


@runtime_checkable
class Scorer(Protocol):
    """判分器契约：输入任务与轨迹，输出分数报告。"""

    async def score(self, case: TaskCase, trajectory: Trajectory) -> ScoreReport:
        ...
