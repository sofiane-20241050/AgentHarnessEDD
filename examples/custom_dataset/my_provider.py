"""15 分钟接入你自己的业务数据集 —— 工单系统（helpdesk 域）示例。

一个数据集 = 一个 DatasetProvider（任务）+ 一个 Environment（工具与虚拟数据库）。
本文件实现一个最小可跑的"工单系统"域：查询 / 关闭 / 升级工单 + 两个任务。

核心原则：数据适配框架，而非框架适配数据。你的业务形态（REST 客户端 / SQL /
内存 dict）都可以，只要满足契约：tools() 返回 ToolDefinition 列表（JSON Schema +
异步函数），reset()/snapshot()/diff() 保证确定性与可断言（错误以返回值回流而非异常，
模型才有机会自恢复——错误恢复也是被测能力）。

运行::

    python my_provider.py      # 自测：加载注册 + 调一次工具
    ahedd datasets list        # import 后出现 helpdesk
    ahedd run --dataset helpdesk --adapter openai-loop
    ahedd score --runs runs/helpdesk --dataset helpdesk

把本文件拷进你的项目 import 即注册；自定义任务要点：
  - rubrics 一条一个可判定约束（判分器逐条核对，支持部分得分）
  - rules.forbidden_tools 可声明业务上不允许调用的工具（确定性通道直接判违例）
  - 需要多轮对话时给 user_scenario 填 persona/traits 并配置用户模拟器
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from ahedd.datasets import register_dataset
from ahedd.datasets.base import TaskCase, TrajectoryRules
from ahedd.env.tools import ToolDefinition

_DB: dict[str, Any] = {
    "tickets": {
        "T001": {"id": "T001", "title": "登录失败", "status": "open", "priority": "P2"},
        "T002": {"id": "T002", "title": "导出报表超时", "status": "open", "priority": "P3"},
    }
}


class HelpdeskEnvironment:
    """实现 Environment 契约：tools / reset / snapshot / diff。"""

    domain = "helpdesk"

    def __init__(self) -> None:
        self.db = copy.deepcopy(_DB)

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_tickets",
                description="按关键词检索工单",
                parameters={
                    "type": "object",
                    "properties": {"keyword": {"type": "string", "description": "标题关键词"}},
                    "required": ["keyword"],
                },
                func=self._search,
            ),
            ToolDefinition(
                name="close_ticket",
                description="关闭工单（只有 open 状态可关闭）",
                parameters={
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                    "required": ["ticket_id"],
                },
                func=self._close,
            ),
            ToolDefinition(
                name="escalate_ticket",
                description="把工单升级为 P1 并通知值班工程师",
                parameters={
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                    "required": ["ticket_id"],
                },
                func=self._escalate,
            ),
        ]

    async def reset(self, seed: int | None = None) -> None:
        self.db = copy.deepcopy(_DB)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.db)

    def diff(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        from ahedd.env.base import default_diff

        return default_diff(before, after)

    # ---- 工具实现（业务规则拒绝以 ok=False 返回，不抛异常） ----

    async def _search(self, keyword: str) -> Any:
        hits = [t for t in self.db["tickets"].values() if keyword in t["title"]]
        return {"ok": True, "count": len(hits), "tickets": hits}

    async def _close(self, ticket_id: str) -> Any:
        ticket = self.db["tickets"].get(ticket_id)
        if ticket is None:
            return {"ok": False, "error": f"ticket not found: {ticket_id}"}
        if ticket["status"] != "open":
            return {"ok": False, "error": f"cannot close ticket in status {ticket['status']}"}
        ticket["status"] = "closed"
        return {"ok": True, "ticket_id": ticket_id, "status": "closed"}

    async def _escalate(self, ticket_id: str) -> Any:
        ticket = self.db["tickets"].get(ticket_id)
        if ticket is None:
            return {"ok": False, "error": f"ticket not found: {ticket_id}"}
        ticket["priority"] = "P1"
        return {"ok": True, "ticket_id": ticket_id, "priority": "P1", "notified": True}


_CASES = [
    TaskCase(
        id="hd_001_close_login_issue",
        domain="helpdesk",
        instruction="帮我关掉那个登录失败的工单。",
        rubrics=["工单 T001 的 status 最终为 closed", "未关闭其他工单"],
        source="custom",
    ),
    TaskCase(
        id="hd_002_escalate_report_timeout",
        domain="helpdesk",
        instruction="导出报表超时这个工单比较急，帮我升级处理。",
        rubrics=["工单 T002 的 priority 最终为 P1"],
        rules=TrajectoryRules(forbidden_tools=["close_ticket"]),
        source="custom",
    ),
]


class HelpdeskProvider:
    name = "helpdesk"

    def domains(self) -> list[str]:
        return ["helpdesk"]

    def load(self, domain: str) -> list[TaskCase]:
        return list(_CASES)

    def build_environment(self, domain: str) -> HelpdeskEnvironment:
        return HelpdeskEnvironment()


register_dataset("helpdesk")(lambda: HelpdeskProvider())

if __name__ == "__main__":
    env = HelpdeskEnvironment()
    result = asyncio.run(env.tools()[1].func(ticket_id="T001"))
    print("self-test close_ticket(T001):", result)
    print("env after:", env.db["tickets"]["T001"])
