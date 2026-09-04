"""MCP server 模块测试：schema 透传 / 参数校验 / 事件日志 / 环境终态。"""


import pytest

from ahedd.datasets import get_dataset
from ahedd.env.tools import ToolDefinition
from ahedd.mcp.server import ServerEventLog, _make_mcp_tool, build_env_server, read_server_events

pytest.importorskip("mcp")


def _mock_env():
    return get_dataset("mock").build_environment("mock")


async def test_server_tools_and_events(tmp_path) -> None:
    env = _mock_env()
    events = tmp_path / "ev.jsonl"
    server = build_env_server(env, events_path=str(events))

    tools = server._tool_manager.list_tools()
    assert sorted(t.name for t in tools) == ["cancel_order", "get_order", "update_address"]
    # JSON Schema 精确透传（非签名推断）
    update = next(t for t in tools if t.name == "update_address")
    assert "new_address" in update.parameters.get("properties", {})

    result = await server._tool_manager.call_tool(
        "update_address", {"order_id": "ORD_1", "new_address": "北京市朝阳区"}
    )
    assert result["ok"] is True

    steps, initial, final = read_server_events(events)
    assert [(s.type, s.tool_name) for s in steps] == [
        ("tool_call", "update_address"),
        ("tool_result", "update_address"),
    ]
    assert initial["orders"]["ORD_1"]["address"] == "上海市浦东新区"
    assert final["orders"]["ORD_1"]["address"] == "北京市朝阳区"


async def test_server_schema_validation(tmp_path) -> None:
    """缺必填参数被我们的 schema 模型拦截（幻觉参数防线）。"""
    env = _mock_env()
    server = build_env_server(env, events_path=str(tmp_path / "ev.jsonl"))
    with pytest.raises(Exception, match="(?i)tool ?error|validation"):
        await server._tool_manager.call_tool("update_address", {"order_id": "ORD_1"})


async def test_error_event_with_classification(tmp_path) -> None:
    """工具执行异常写入 error 事件并带分类。"""
    env = _mock_env()
    events = tmp_path / "ev.jsonl"
    log = ServerEventLog(events)

    async def bad(**kwargs):
        raise RuntimeError("business rule violation")

    tool = ToolDefinition(
        name="bad_tool", description="会炸", parameters={"type": "object", "properties": {}}, func=bad
    )
    mcp_tool = _make_mcp_tool(tool, env, log)
    with pytest.raises(Exception, match="business rule violation"):  # ToolError 包装后抛出
        await mcp_tool.run({})

    steps, _, _ = read_server_events(events)
    error = next(s for s in steps if s.type == "error")
    assert error.tool_name == "bad_tool"
    assert error.error_kind == "tool"  # RuntimeError 业务异常归 tool 类
