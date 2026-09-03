"""错误分类：区分 Agent 自身错误与基础设施错误（调研报告 §6.4 归因分类学的错误面）。

CANONICAL_ERROR_KINDS:
  tool  工具执行失败（业务规则拒绝、数据不存在等环境内错误）
  agent Agent 自身错误（幻觉工具、参数与 schema 不匹配、超轮次、死循环）
  infra 基础设施错误（网络波动、超时、限流 429、5xx、并发打满）——不应计入 Agent 失败

按异常类型的继承链（MRO）名称判定，不直接 import 依赖，可选车道（MCP/httpx）同样适用。
"""

from __future__ import annotations

CANONICAL_ERROR_KINDS = ("tool", "agent", "infra")

_INFRA_MARKERS = (
    "APIError", "APIConnectionError", "APITimeoutError", "APIStatusError",
    "RateLimitError", "InternalServerError", "HTTPError", "NetworkError",
    "TimeoutError", "ConnectError", "ReadError", "RemoteProtocolError",
)
_AGENT_MARKERS = ("TypeError", "KeyError")  # 参数与签名/schema 不匹配的典型异常


def classify_exception(exc: BaseException) -> str:
    """按异常 MRO 名称分类为 infra / agent / tool（默认 tool：环境工具内的业务异常）。"""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & set(_INFRA_MARKERS):
        return "infra"
    if names & set(_AGENT_MARKERS):
        return "agent"
    return "tool"
