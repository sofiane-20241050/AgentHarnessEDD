"""rubric 滑窗判分器测试：FakeLLM 作 judge，验证粘滞状态与鲁棒解析。"""

import pytest

from ahedd.datasets.base import TaskCase
from ahedd.llm.fake import FakeLLMClient
from ahedd.scoring.rubric import RubricSlidingWindowScorer, _parse_results
from ahedd.trace.schema import RunMeta, StepRecord, Trajectory


def _case() -> TaskCase:
    return TaskCase(
        id="t1",
        domain="mock",
        instruction="x",
        rubrics=["订单 ORD_1 地址最终为 北京市朝阳区", "Agent 向用户确认了结果"],
    )


def _traj() -> Trajectory:
    return Trajectory(
        meta=RunMeta(task_id="t1", domain="mock"),
        steps=[
            StepRecord(index=0, type="user", content="改地址"),
            StepRecord(index=1, type="assistant", content="好的"),
            StepRecord(
                index=2, type="tool_call", tool_name="update_address",
                tool_args={"order_id": "ORD_1", "new_address": "北京市朝阳区"},
            ),
            StepRecord(index=3, type="tool_result", tool_name="update_address",
                       tool_result={"ok": True, "address": "北京市朝阳区"}),
            StepRecord(index=4, type="assistant", content="已改好", stop_reason="stop"),
        ],
    )


def _judge_resp(results: list[dict]) -> dict:
    import json

    return {"content": json.dumps({"results": results}, ensure_ascii=False)}


async def test_single_window_scoring() -> None:
    judge = FakeLLMClient([
        _judge_resp([
            {"key": "r0", "satisfied": True, "evidence_turn": 1, "rationale": "工具调用与返回证实"},
            {"key": "r1", "satisfied": False, "evidence_turn": None, "rationale": "无确认"},
        ])
    ])
    scorer = RubricSlidingWindowScorer(judge)
    report = await scorer.score(_case(), _traj())
    assert report.score == pytest.approx(0.5)
    assert report.passed is False
    assert report.rubric_results[0].satisfied is True
    assert report.rubric_results[0].evidence_turn == 1
    assert report.rubric_results[1].satisfied is False
    assert report.rule_violations == []


async def test_sticky_state_across_windows() -> None:
    """已满足的 rubric 不因后续窗口撤销（粘滞），跨窗口传播。"""
    judge = FakeLLMClient([
        _judge_resp([{"key": "r0", "satisfied": True, "evidence_turn": 1, "rationale": "w1"}]),
        _judge_resp([
            {"key": "r0", "satisfied": False, "evidence_turn": None, "rationale": "试图撤销"},
            {"key": "r1", "satisfied": False, "evidence_turn": None, "rationale": "w2"},
        ]),
        _judge_resp([{"key": "r1", "satisfied": True, "evidence_turn": 3, "rationale": "w3"}]),
    ])
    scorer = RubricSlidingWindowScorer(judge, window_turns=1, overlap_turns=0)
    report = await scorer.score(_case(), _traj())
    assert report.rubric_results[0].satisfied is True  # 粘滞：w2 的 false 不生效
    assert report.rubric_results[1].satisfied is True
    assert report.passed is True
    assert report.score == 1.0


def test_parse_results_robust() -> None:
    fenced = '废话```json\n{"results": [{"key": "r0", "satisfied": true}]}\n```'
    assert _parse_results(fenced)[0]["key"] == "r0"
    with_think = "<think>让我想想</think>\n{\"results\": []}"
    assert _parse_results(with_think) == []
    assert _parse_results("完全不是 JSON") == []
