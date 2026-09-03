"""被测 Agent 适配器注册机制。

适配器以工厂函数形式注册：name -> () -> AgentAdapter。
内置适配器在各模块文件底部自注册；第三方适配器调用 register_adapter()。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahedd.adapters.base import AgentAdapter

AdapterFactory = Callable[[], "AgentAdapter"]

_REGISTRY: dict[str, AdapterFactory] = {}


def register_adapter(name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    def _wrap(factory: AdapterFactory) -> AdapterFactory:
        _REGISTRY[name] = factory
        return factory

    return _wrap


def get_adapter(name: str) -> AgentAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown adapter: {name!r}, available: {sorted(_REGISTRY)}; "
            "第三方适配器需先 import 其注册模块"
        )
    return _REGISTRY[name]()


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def _load_builtin_adapters() -> None:
    """导入内置适配器模块以触发自注册（可延迟、可失败：按需安装 extra）。"""
    from importlib import import_module

    for mod in ("ahedd.adapters.openai_loop", "ahedd.adapters.deepagents_adapter", "ahedd.adapters.tau_adapter"):
        try:
            import_module(mod)
        except ImportError:
            continue  # 对应 extra 未安装，属预期


_load_builtin_adapters()
