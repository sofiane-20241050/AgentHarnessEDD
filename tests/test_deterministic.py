"""确定性断言通道测试：禁调工具 / 轮次上限 / 死循环 / 失败恢复观测。"""

from ahedd.datasets.base import TaskCase, TrajectoryRules
from ahedd.scoring.deterministic import check_trajectory_rules, count_repeated_failures
from ahedd.trace.schema import RunMeta, StepRecord, Trajectory


def _case(**rules) -> TaskCase:
    return TaskCase(
        id="t1",
        domain="delivery",
        instruction="x",
        rules=TrajectoryRules(**rules),
    )


def _traj(steps: list[StepRecord]) -> Trajectory:
    return Trajectory(meta=RunMeta(task_id="t1"), steps=steps)


def _s(index: int, type: str, **kw) -> StepRecord:
    return StepRecord(index=index, type=type, **kw)  # type: ignore[arg-type]


def test_forbidden_tool_detected() -> None:
    steps = [
        _s(0, "assistant", content="ok"),
        _s(1, "tool_call", tool_name="kill_switch", tool_args={}),
    ]
    v = check_trajectory_rules(_case(forbidden_tools=["kill_switch"]), _traj(steps))
    assert any("forbidden" in x for x in v)


def test_loop_detected() -> None:
    steps = [
        _s(i, "tool_call", tool_name="search", tool_args={"q": "same"})
        for i in range(4)
    ]
    v = check_trajectory_rules(_case(max_identical_calls=3), _traj(steps))
    assert any("loop" in x for x in v)


def test_max_turns() -> None:
    steps = [_s(i, "assistant", content="m") for i in range(10)]
    v = check_trajectory_rules(_case(max_turns=3), _traj(steps))
    assert any("max_turns" in x for x in v)


def test_clean_trajectory_passes() -> None:
    steps = [
        _s(0, "assistant", content="hi"),
        _s(1, "tool_call", tool_name="search", tool_args={"q": "a"}),
        _s(2, "tool_result", tool_name="search", tool_result=[1]),
        _s(3, "assistant", content="done"),
    ]
    assert check_trajectory_rules(_case(), _traj(steps)) == []


def test_repeated_failures_observed() -> None:
    steps = [
        _s(0, "tool_call", tool_name="pay", tool_args={}),
        _s(1, "error", content="E1"),
        _s(2, "tool_call", tool_name="pay", tool_args={}),
        _s(3, "error", content="E2"),
        _s(4, "assistant", content="give up"),
    ]
    assert count_repeated_failures(steps) == 2
