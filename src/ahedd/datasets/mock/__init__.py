"""内置 mock 数据集：框架自测域（不依赖任何外部数据与 API）。

作用与 τ²-bench 的 mock 域相同：用最小环境验证框架机制本身——
轨迹录制、写操作 diff、确定性断言（禁调/死循环）、runner 端到端。
"""

from __future__ import annotations

import copy
from typing import Any

from ahedd.datasets import register_dataset
from ahedd.datasets.base import TaskCase, TrajectoryRules
from ahedd.env.base import default_diff
from ahedd.env.tools import ToolDefinition

_INITIAL_DB: dict[str, Any] = {
    "orders": {
        "ORD_1": {"id": "ORD_1", "status": "SHIPPED", "address": "上海市浦东新区"},
    }
}


class MockEnvironment:
    """订单域迷你环境：1 张订单表 + 3 个工具（读 / 写 / 会失败的业务规则）。"""

    domain = "mock"

    def __init__(self) -> None:
        self.db: dict[str, Any] = copy.deepcopy(_INITIAL_DB)

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_order",
                description="查询订单详情",
                parameters={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                func=self._get_order,
            ),
            ToolDefinition(
                name="update_address",
                description="修改订单配送地址",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "new_address": {"type": "string"},
                    },
                    "required": ["order_id", "new_address"],
                },
                func=self._update_address,
            ),
            ToolDefinition(
                name="cancel_order",
                description="取消订单。仅未发货（PENDING）订单可取消。",
                parameters={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                func=self._cancel_order,
            ),
        ]

    async def reset(self, seed: int | None = None) -> None:
        self.db = copy.deepcopy(_INITIAL_DB)

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.db)

    def diff(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return default_diff(before, after)

    # ---- 工具实现（确定性 Python 函数） ----

    async def _get_order(self, order_id: str) -> dict[str, Any]:
        order = self.db["orders"].get(order_id)
        return order if order else {"ok": False, "error": f"order not found: {order_id}"}

    async def _update_address(self, order_id: str, new_address: str) -> dict[str, Any]:
        order = self.db["orders"].get(order_id)
        if not order:
            return {"ok": False, "error": f"order not found: {order_id}"}
        order["address"] = new_address
        return {"ok": True, "order_id": order_id, "address": new_address}

    async def _cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self.db["orders"].get(order_id)
        if not order:
            return {"ok": False, "error": f"order not found: {order_id}"}
        if order["status"] != "PENDING":
            return {"ok": False, "error": f"cannot cancel order in status {order['status']}"}
        order["status"] = "CANCELLED"
        return {"ok": True, "order_id": order_id, "status": "CANCELLED"}


_CASES: list[TaskCase] = [
    TaskCase(
        id="mock_001_change_address",
        domain="mock",
        instruction="帮我把订单 ORD_1 的收货地址改成 北京市朝阳区。",
        rubrics=["订单 ORD_1 的配送地址最终为 北京市朝阳区"],
        source="builtin",
    ),
    TaskCase(
        id="mock_002_reject_cancel_shipped",
        domain="mock",
        instruction="帮我把订单 ORD_1 取消了。",
        rubrics=["订单 ORD_1 的 status 保持 SHIPPED 未被取消", "Agent 向用户说明了无法取消的原因"],
        source="builtin",
    ),
    TaskCase(
        id="mock_003_loop_detection",
        domain="mock",
        instruction="查询订单 ORD_1。（配合脚本重放制造死循环，用于验证确定性断言）",
        rubrics=["任意"],
        rules=TrajectoryRules(max_identical_calls=3, max_turns=50),
        source="builtin",
    ),
]


class MockProvider:
    name = "mock"

    def domains(self) -> list[str]:
        return ["mock"]

    def load(self, domain: str) -> list[TaskCase]:
        return list(_CASES)

    def build_environment(self, domain: str) -> MockEnvironment:
        return MockEnvironment()


@register_dataset("mock")
def _factory() -> MockProvider:
    return MockProvider()
