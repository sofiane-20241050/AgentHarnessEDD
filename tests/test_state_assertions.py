"""终态断言通道测试 + judge prompt 官方要素。"""

from ahedd.datasets.base import TaskCase
from ahedd.llm.fake import FakeLLMClient
from ahedd.scoring.deterministic import check_expected_states
from ahedd.scoring.rubric import RubricSlidingWindowScorer
from ahedd.trace.schema import RunMeta, StepRecord, Trajectory


def test_check_expected_states_pass_and_fail():
    expected_states = [
        {"required_orders": [
            {"order_id": "T001", "status": "processed", "total_price": 25},
        ]},
    ]
    ok_state = {"orders": {"T001": {"order_id": "T001", "status": "processed", "total_price": 25, "note": ""}}}
    assert check_expected_states(expected_states, ok_state) == []

    bad_price = {"orders": {"T001": {"order_id": "T001", "status": "processed", "total_price": 94}}}
    violations = check_expected_states(expected_states, bad_price)
    assert any("total_price" in v for v in violations)

    missing = {"orders": {}}
    assert any("not found" in v for v in check_expected_states(expected_states, missing))

    # orders 为 list 形态（MCP 事件快照兼容）
    list_state = {"orders": [{"order_id": "T001", "status": "processed", "total_price": 25}]}
    assert check_expected_states(expected_states, list_state) == []


async def test_judge_prompt_contains_official_elements(tmp_path):
    """judge prompt 含任务指令与环境时间（吸收 VitaBench 官方 evaluator 要素）。"""
    case = TaskCase(
        id="t1", domain="mock", instruction="帮我点份清淡米线",
        rubrics=["订单已生成"],
        extra={"env_time": "2025-06-21 11:20:00 星期六"},
    )
    traj = Trajectory(meta=RunMeta(task_id="t1"), steps=[
        StepRecord(index=0, type="user", content="点米线"),
        StepRecord(index=1, type="tool_call", tool_name="create_order", tool_args={}),
        StepRecord(index=2, type="tool_result", tool_name="create_order",
                   tool_result={"ok": True, "order_id": "T001"}),
        StepRecord(index=3, type="assistant", content="已下单", stop_reason="stop"),
    ])
    captured = {}

    class SpyClient(FakeLLMClient):
        async def chat(self, messages, tools=None, **kwargs):
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            from ahedd.llm.base import LLMResponse

            return LLMResponse(
                content='{"results": [{"key": "r0", "satisfied": true, "evidence_turn": 1, "rationale": "工具返回确认"}]}'
            )

    scorer = RubricSlidingWindowScorer(SpyClient())
    report = await scorer.score(case, traj)
    assert report.passed is True
    assert "帮我点份清淡米线" in captured["user"]          # 任务指令入 prompt
    assert "2025-06-21 11:20:00" in captured["user"]      # 环境时间入 prompt
    assert "订单类 rubric" in captured["system"]            # 官方订单确认硬规则
