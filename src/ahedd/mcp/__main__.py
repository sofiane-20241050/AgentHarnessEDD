"""`python -m ahedd.mcp` 命令行入口。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="ahedd-mcp", description="把数据集环境暴露为 MCP Server")
    parser.add_argument("--dataset", default="mock", help="数据集名（ahedd datasets list）")
    parser.add_argument("--domain", default=None, help="域（缺省取第一个）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stdio", action="store_true", help="stdio 传输（默认）")
    mode.add_argument("--http", action="store_true", help="streamable-http 传输")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--events-file", default=None, help="工具/环境事件日志（JSONL），供轨迹合并")
    args = parser.parse_args()

    from ahedd.mcp.server import run_server

    run_server(
        args.dataset,
        domain=args.domain,
        transport="streamable-http" if args.http else "stdio",
        host=args.host,
        port=args.port,
        events_path=args.events_file,
    )


if __name__ == "__main__":
    main()
