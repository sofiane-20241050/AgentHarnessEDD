"""车道二：把评测环境暴露为 MCP Server，供外部/闭源 Agent 接入。

方向说明：不是我们去连别人的工具，而是让被测 Agent（Claude Code、
Codex CLI、Dify 等 MCP 客户端）连进我们的仿真世界——它的每次工具
调用都发生在沙箱内，可录制、可断言、可回滚。

安装：pip install "agentharness-edd[mcp]"。D2+ 里程碑实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahedd.env.base import Environment


async def serve_env_as_mcp(env: Environment, transport: str = "stdio") -> None:
    """将 Environment 的工具集发布为 MCP Server。

    TODO(D2+)：
      - env.tools() -> MCP tool 声明（name/description/inputSchema 直接同构）
      - 调用经 TrajectoryRecorder.wrap_tool 包装后执行
      - transport: stdio（本地 CLI 型 Agent）/ streamable-http（远程平台）
    """
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ImportError('MCP 车道需要额外依赖：pip install "agentharness-edd[mcp]"') from exc
    raise NotImplementedError("MCP Server 于 D2+ 里程碑实现")
