"""数据集层入口：注册机制 + 基类再导出 + 内置数据集加载。

第三方/私有数据集：实现 DatasetProvider 后调用 register_dataset()。
"""

from __future__ import annotations

from collections.abc import Callable

from ahedd.datasets.base import DatasetProvider, TaskCase, TrajectoryRules, UserScenario

ProviderFactory = Callable[[], "DatasetProvider"]

_REGISTRY: dict[str, ProviderFactory] = {}


def register_dataset(name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    def _wrap(factory: ProviderFactory) -> ProviderFactory:
        _REGISTRY[name] = factory
        return factory

    return _wrap


def get_dataset(name: str) -> DatasetProvider:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown dataset: {name!r}, available: {sorted(_REGISTRY)}; "
            "私有数据集请实现 DatasetProvider 后调用 register_dataset()"
        )
    return _REGISTRY[name]()


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


def _load_builtin_datasets() -> None:
    """导入内置数据集模块以触发自注册（mock 为框架自测域，vita 于 D1 接入）。"""
    from importlib import import_module

    for mod in ("ahedd.datasets.mock", "ahedd.datasets.vita"):
        try:
            import_module(mod)
        except ImportError:
            continue


_load_builtin_datasets()

__all__ = [
    "DatasetProvider",
    "TaskCase",
    "TrajectoryRules",
    "UserScenario",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
