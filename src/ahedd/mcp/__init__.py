"""MCP 模块入口：环境作为 MCP Server 供给任意 MCP 客户端被测对象。

用法::

    # stdio（本地宿主，如 LangChain 系）
    python -m ahedd.mcp --dataset mock --stdio

    # streamable-http（远程宿主，配合 ssh -R 反向隧道）
    python -m ahedd.mcp --dataset mock --http --port 8023 --events-file runs/mcp_events.jsonl
"""

from ahedd.mcp.server import (
    ServerEventLog,
    build_env_server,
    read_server_events,
    run_server,
)

__all__ = ["ServerEventLog", "build_env_server", "read_server_events", "run_server"]
