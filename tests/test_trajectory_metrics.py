"""轨迹动力学指标测试。"""

from ahedd.scoring.trajectory_metrics import compute_trajectory_metrics, summarize_suite
from ahedd.trace.schema import RunMeta, StepRecord, Trajectory, Usage


def _traj(steps: list[StepRecord]) -> Trajectory:
    meta = RunMeta(task_id="t1")
    meta.total_usage = Usage(input_tokens=100, output_tokens=20)
    return Trajectory(meta=meta, steps=steps)


def _s(index: int, type: str, **kw):
    return StepRecord(index=index, type=type, **kw)  # type: ignore[arg-type]


def test_useful_ratio_and_errors() -> None:
    steps = [
        _s(0, "user", content="go"),
        _s(1, "tool_call", tool_name="get_order", tool_args={"order_id": "A"}),
        _s(2, "tool_result", tool_name="get_order", tool_result={"ok": True}),
        _s(3, "tool_call", tool_name="get_order", tool_args={"order_id": "A"}),  # 冗余重复
        _s(4, "tool_result", tool_name="get_order", tool_result={"ok": True}),
        _s(5, "tool_call", tool_name="cancel_order", tool_args={"order_id": "A"}),
        _s(6, "tool_result", tool_name="cancel_order", tool_result={"ok": False, "error": "shipped"}),
        _s(7, "error", content="unknown tool: hack"),
        _s(8, "assistant", content="done", stop_reason="stop"),
    ]
    m = compute_trajectory_metrics(_traj(steps))
    assert m.tool_calls == 3
    assert m.distinct_tool_calls == 2
    assert m.repeated_calls == 1
    assert m.useful_action_ratio == round(2 / 3, 4)
    assert m.failed_calls == 1
    assert m.unknown_tool_calls == 1
    assert m.tool_error_rate == round(1 / 4, 4)
    assert m.had_error is True
    assert m.recovered is True  # 出错后仍正常收卷
    assert m.tokens_in == 100 and m.tokens_out == 20


def test_recovery_false_when_ends_badly() -> None:
    steps = [
        _s(0, "user", content="go"),
        _s(1, "error", content="boom"),
        _s(2, "error", content="boom"),
    ]
    m = compute_trajectory_metrics(_traj(steps))
    assert m.max_failure_streak == 2
    assert m.recovered is False


def test_suite_summary() -> None:
    steps_a = [
        _s(0, "user"), _s(1, "assistant", content="ok", stop_reason="stop"),
    ]
    steps_b = [
        _s(0, "user"),
        _s(1, "tool_call", tool_name="x", tool_args={}),
        _s(2, "tool_result", tool_name="x", tool_result={"ok": True}),
        _s(3, "assistant", content="ok", stop_reason="stop"),
    ]
    ma = compute_trajectory_metrics(_traj(steps_a))
    mb = compute_trajectory_metrics(_traj(steps_b))
    summary = summarize_suite([
        ("openai-loop", "c1", True, ma),
        ("openai-loop", "c2", False, mb),
    ])
    a = summary["adapters"]["openai-loop"]
    assert a["runs"] == 2 and a["scored"] == 2
    assert a["pass_rate"] == 0.5
    assert a["sts_median_turns"] == 1  # 成功轨迹（c1）的回合数
    assert a["pareto"]["avg_tokens_per_run"] == 120.0  # 两条轨迹 (100+20)*2 / 2


def test_error_classification_infra_excluded() -> None:
    """基础设施错误（网络/限流）不计入 Agent 失败：恢复率、连击、错误率全部豁免。"""
    from ahedd.trace.errors import classify_exception

    assert classify_exception(TimeoutError("t")) == "infra"
    assert classify_exception(TypeError("x")) == "agent"
    assert classify_exception(ValueError("v")) == "tool"

    steps = [
        _s(0, "user", content="go"),
        _s(1, "tool_call", tool_name="pay", tool_args={}),
        _s(2, "error", content="APITimeoutError: boom", error_kind="infra"),
        _s(3, "error", content="APITimeoutError: boom2", error_kind="infra"),
        _s(4, "assistant", content="done", stop_reason="stop"),
    ]
    m = compute_trajectory_metrics(_traj(steps))
    assert m.error_events == 2
    assert m.infra_errors == 2
    assert m.agent_errors == 0
    assert m.had_agent_error is False
    assert m.recovered is None  # 无 Agent 错误时无意义
    assert m.max_failure_streak == 0  # infra 不计入连击
    assert m.tool_error_rate == 0.0  # infra 不进错误率


def test_error_classification_mixed_streak() -> None:
    """infra 错误夹在中间不打断 Agent 的失败连击。"""
    steps = [
        _s(0, "user", content="go"),
        _s(1, "error", content="TypeError: bad args", error_kind="agent"),
        _s(2, "error", content="ConnectError: net", error_kind="infra"),
        _s(3, "error", content="ValueError: business", error_kind="tool"),
        _s(4, "assistant", content="ok", stop_reason="stop"),
    ]
    m = compute_trajectory_metrics(_traj(steps))
    assert m.agent_errors == 2  # agent + tool（非 infra 归 Agent 侧）
    assert m.infra_errors == 1
    assert m.max_failure_streak == 2  # infra 被跳过，两侧连击
    assert m.recovered is True
