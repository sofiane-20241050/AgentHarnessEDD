"""Claude Code 适配器离线测试：文本协议工具桥 + resume 续轮 + 用量累计。"""

import json

from ahedd.adapters.claude_code_adapter import ClaudeCodeAdapter, _extract_tool_call
from ahedd.datasets import get_dataset
from ahedd.runner import run_case


def _cc_result(text: str, in_tok: int = 100, out_tok: int = 20) -> dict:
    return {
        "type": "result",
        "is_error": False,
        "result": text,
        "session_id": "s-1",
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "total_cost_usd": 0.01,
    }


async def test_e2e_text_protocol_bridge(tmp_path, monkeypatch) -> None:
    provider = get_dataset("mock")
    case = next(c for c in provider.load("mock") if c.id == "mock_001_change_address")
    script = [
        _cc_result(
            '```json\n{"tool_call": {"name": "update_address", '
            '"arguments": {"order_id": "ORD_1", "new_address": "北京市朝阳区"}}}\n```'
        ),
        _cc_result("已为您把订单 ORD_1 的收货地址改为 北京市朝阳区"),
    ]
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_call(self, prompt, session_id, append_system):
        calls.append((prompt, session_id, append_system))
        return script[len(calls) - 1]

    monkeypatch.setattr(ClaudeCodeAdapter, "_call_claude", fake_call)
    adapter = ClaudeCodeAdapter(workdir="/tmp/x", ssh_target=None)
    env = provider.build_environment("mock")

    outcome = await run_case(
        dataset="mock",
        adapter=adapter,
        env=env,
        task_id=case.id,
        instruction=case.instruction,
        trace_dir=str(tmp_path),
    )

    assert outcome.stop_reason == "stop"
    types = [s.type for s in outcome.trajectory.steps]
    assert types == ["user", "assistant", "tool_call", "tool_result", "assistant"]
    assert "北京市朝阳区" in json.dumps(outcome.env_diff, ensure_ascii=False)
    # 首轮带工具目录附加系统提示；次轮 resume 同一会话且不带附加段
    assert "update_address" in (calls[0][2] or "")
    assert calls[1][1] == "s-1"
    assert calls[1][2] is None
    # 工具结果以 [TOOL RESULT] 回传
    assert calls[1][0].startswith("[TOOL RESULT]")
    # 用量累计（CC 每轮 usage 汇总）
    assert outcome.trajectory.meta.total_usage.input_tokens == 200
    assert outcome.trajectory.meta.total_usage.output_tokens == 40


def test_extract_tool_call() -> None:
    fenced = '好的\n```json\n{"tool_call": {"name": "x", "arguments": {"a": 1}}}\n```'
    assert _extract_tool_call(fenced) == ("x", {"a": 1})
    assert _extract_tool_call("普通文本，没有调用") is None
    assert _extract_tool_call("```json\n{\"broken\": true}\n```") is None
