"""端到端测试：mock 数据集 + FakeLLM 离线验证核心链路。

链路：DatasetProvider -> Environment -> recorder 包装工具 -> OpenAILoopAdapter
     -> Trajectory 落盘 -> 确定性断言 -> env diff。全程零 API 成本。
"""

import json
from pathlib import Path

import pytest

from ahedd.adapters.openai_loop import OpenAILoopAdapter
from ahedd.datasets import get_dataset, list_datasets
from ahedd.llm.fake import FakeLLMClient
from ahedd.runner import run_case
from ahedd.scoring.deterministic import check_trajectory_rules
from ahedd.trace.schema import load_jsonl_trajectory


@pytest.fixture()
def provider():
    assert "mock" in list_datasets()
    return get_dataset("mock")


def _case(provider, case_id: str):
    return next(c for c in provider.load("mock") if c.id == case_id)


async def test_e2e_change_address(tmp_path: Path, provider) -> None:
    """正常路径：查单 -> 改地址 -> 汇报；写操作体现在 env diff 与轨迹。"""
    script = [
        {"tool_calls": [{"id": "c1", "name": "get_order", "args": {"order_id": "ORD_1"}}]},
        {
            "tool_calls": [
                {"id": "c2", "name": "update_address", "args": {"order_id": "ORD_1", "new_address": "北京市朝阳区"}}
            ]
        },
        {"content": "已为您把订单 ORD_1 的收货地址改为 北京市朝阳区。", "finish_reason": "stop"},
    ]
    env = provider.build_environment("mock")
    case = _case(provider, "mock_001_change_address")
    adapter = OpenAILoopAdapter(FakeLLMClient(script))

    outcome = await run_case(
        dataset="mock",
        adapter=adapter,
        env=env,
        task_id=case.id,
        instruction=case.instruction,
        trace_dir=str(tmp_path),
    )

    assert outcome.stop_reason == "stop"
    assert "北京市朝阳区" in outcome.final_message
    # 写操作体现为环境状态 diff
    orders_diff = outcome.env_diff["orders"]
    assert "北京市朝阳区" in json.dumps(orders_diff, ensure_ascii=False)
    # 轨迹事件顺序：user -> (assistant, tool_call, tool_result) x2 -> assistant(终答)
    types = [s.type for s in outcome.trajectory.steps]
    assert types == [
        "user", "assistant", "tool_call", "tool_result",
        "assistant", "tool_call", "tool_result", "assistant",
    ]
    # 用量：Fake 每次补全 1 in / 1 out，共 3 次；逐步记录 + meta 汇总
    assistant_steps = [s for s in outcome.trajectory.steps if s.type == "assistant"]
    assert all(s.usage.input_tokens == 1 and s.usage.output_tokens == 1 for s in assistant_steps)
    assert outcome.trajectory.meta.total_usage.input_tokens == 3
    assert outcome.trajectory.meta.total_usage.output_tokens == 3
    # ISO 8601 时间戳（人类可读，与 started_at 同格式）
    assert outcome.trajectory.steps[0].ts.startswith("20")
    # token 计量累计（Fake 每步 1/1，共 3 次补全）
    assert outcome.trajectory.meta.adapter == "openai-loop"
    # 确定性断言通过
    assert check_trajectory_rules(case, outcome.trajectory) == []
    # 轨迹落盘可读回（先存轨迹后判分的前提）
    trace_file = tmp_path / "mock" / "mock" / case.id / f"{outcome.trajectory.meta.run_id}.jsonl"
    reloaded = load_jsonl_trajectory(str(trace_file))
    assert len(reloaded.steps) == len(outcome.trajectory.steps)


async def test_e2e_reject_cancel_shipped(tmp_path: Path, provider) -> None:
    """业务规则拒绝：取消已发货订单必须失败，终态保持 SHIPPED。"""
    script = [
        {"tool_calls": [{"id": "c1", "name": "cancel_order", "args": {"order_id": "ORD_1"}}]},
        {"content": "抱歉，订单已发货，无法取消。", "finish_reason": "stop"},
    ]
    env = provider.build_environment("mock")
    case = _case(provider, "mock_002_reject_cancel_shipped")
    adapter = OpenAILoopAdapter(FakeLLMClient(script))

    outcome = await run_case(
        dataset="mock",
        adapter=adapter,
        env=env,
        task_id=case.id,
        instruction=case.instruction,
        trace_dir=str(tmp_path),
    )

    # 终态未被篡改：env diff 为空（取消被业务规则拒绝）
    assert outcome.env_diff == {}
    tool_result = outcome.trajectory.steps[3]
    assert tool_result.type == "tool_result"
    assert tool_result.tool_result["ok"] is False


async def test_e2e_loop_detected(tmp_path: Path, provider) -> None:
    """死循环检测：同一工具同参数连续重复触发确定性违例。"""
    repeated = [{"id": f"c{i}", "name": "get_order", "args": {"order_id": "ORD_1"}} for i in range(4)]
    script = [
        *[{"tool_calls": [tc]} for tc in repeated],
        {"content": "done", "finish_reason": "stop"},
    ]
    env = provider.build_environment("mock")
    case = _case(provider, "mock_003_loop_detection")
    adapter = OpenAILoopAdapter(FakeLLMClient(script))

    outcome = await run_case(
        dataset="mock",
        adapter=adapter,
        env=env,
        task_id=case.id,
        instruction=case.instruction,
        trace_dir=str(tmp_path),
    )

    violations = check_trajectory_rules(case, outcome.trajectory)
    assert any("loop" in v for v in violations)
