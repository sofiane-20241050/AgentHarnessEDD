"""环境 -> MCP Server：把任意 Environment 的工具面暴露给 MCP 客户端被测对象。

设计（调研报告 §3.3 车道二 / §4）：
  - 不集成框架，框架来连我们的世界：CC / Codex CLI / dsh / Dify / LangChain 系
    均可作为 MCP 客户端消费同一份环境工具面（控制变量的统一注入层）
  - 工具执行发生在本（server）侧：JSON Schema 经 FuncMetadata 精确透传（非签名推断），
    调用即写事件日志（tool_call/tool_result/error/env_state），供适配器事后合并入统一轨迹
  - env_state 事件（初始 + 每次工具调用后快照）支撑跨进程的环境终态 diff

传输：stdio（本地子进程宿主）/ streamable-http（远程宿主，配合 ssh -R 反向隧道）。

安装：pip install "agentharness-edd[mcp]"（mcp>=1.2,<2）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ahedd.env.tools import ToolDefinition, schema_to_args_model
from ahedd.trace.errors import classify_exception
from ahedd.trace.schema import StepRecord

if TYPE_CHECKING:
    from ahedd.env.base import Environment


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ServerEventLog:
    """MCP server 侧的事件日志（JSONL 追加写）。适配器在 run 结束后读取合并。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        event.setdefault("ts", _now())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def build_env_server(
    env: Environment,
    *,
    server_name: str = "ahedd",
    events_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8023,
) -> Any:
    """把 Environment 的工具注册为 MCP server（返回 mcp.server.fastmcp.FastMCP）。"""
    from mcp.server.fastmcp import FastMCP

    log = ServerEventLog(events_path) if events_path else None
    if log:
        log.write({"type": "env_state", "state": env.snapshot()})

    server = FastMCP(server_name, host=host, port=port)

    for tool in env.tools():
        server._tool_manager._tools[tool.name] = _make_mcp_tool(tool, env, log)
    return server


def _make_mcp_tool(tool: ToolDefinition, env: Environment, log: ServerEventLog | None) -> Any:
    """构造单个 MCP Tool：JSON Schema 精确透传 + 参数经 schema 模型校验 + 事件入日志。"""
    from mcp.server.fastmcp.tools.base import Tool
    from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata

    inner = tool.func
    args_model = schema_to_args_model(tool.name, tool.parameters, bases=(ArgModelBase,))

    async def fn(**kwargs: Any) -> Any:
        if log:
            log.write({"type": "tool_call", "tool_name": tool.name, "tool_args": kwargs})
        try:
            result = await inner(**kwargs)
        except Exception as exc:
            if log:
                log.write(
                    {
                        "type": "error",
                        "tool_name": tool.name,
                        "content": f"{type(exc).__name__}: {exc}",
                        "error_kind": classify_exception(exc),
                    }
                )
            raise
        if log:
            log.write({"type": "tool_result", "tool_name": tool.name, "tool_result": result})
            log.write({"type": "env_state", "state": env.snapshot()})
        return result

    return Tool(
        fn=fn,
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        fn_metadata=FuncMetadata(arg_model=args_model),
        is_async=True,
    )


def read_server_events(path: str | Path) -> tuple[list[StepRecord], dict[str, Any], dict[str, Any]]:
    """读回 server 事件日志 -> (工具/错误 StepRecord 列表, 初始环境快照, 最终环境快照)。"""
    steps: list[StepRecord] = []
    initial_state: dict[str, Any] = {}
    final_state: dict[str, Any] = {}
    index = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        etype = event.get("type")
        if etype == "env_state":
            state = event.get("state") or {}
            if not initial_state:
                initial_state = state
            final_state = state
        elif etype in ("tool_call", "tool_result", "error"):
            steps.append(
                StepRecord(  # type: ignore[call-arg]
                    index=index,
                    type=etype,
                    content=event.get("content", ""),
                    tool_name=event.get("tool_name"),
                    tool_args=event.get("tool_args"),
                    tool_result=event.get("tool_result"),
                    error_kind=event.get("error_kind"),
                    ts=event.get("ts", _now()),
                )
            )
            index += 1
    return steps, initial_state, final_state


def run_server(
    dataset: str,
    *,
    domain: str | None = None,
    transport: str = "stdio",  # stdio | streamable-http
    host: str = "127.0.0.1",
    port: int = 8023,
    events_path: str | Path | None = None,
) -> None:
    """构建并运行一个数据集环境的 MCP server（阻塞）。"""
    from ahedd.datasets import get_dataset

    provider = get_dataset(dataset)
    domain = domain or provider.domains()[0]
    env = provider.build_environment(domain)
    server = build_env_server(env, events_path=events_path, host=host, port=port)
    server.run(transport="stdio" if transport == "stdio" else "streamable-http")
