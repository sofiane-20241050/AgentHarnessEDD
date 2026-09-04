"""工具统一形态与注册表（环境层概念：环境暴露工具，适配器只是消费方）。

ToolDefinition 是 OpenAI function-calling 兼容的三元组（名称/描述/JSON Schema + 异步实现）；
ToolRegistry 提供各车道适配器共用的查找与格式转换，消除每个适配器重复造 by_name 字典。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

JsonSchema = dict[str, Any]


@dataclass
class ToolDefinition:
    """环境工具的统一形态：OpenAI function-calling 兼容。

    parameters 为 JSON Schema；func 为异步可调用，参数名与 schema 一致。
    """

    name: str
    description: str
    parameters: JsonSchema
    func: Callable[..., Awaitable[Any]]


_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class _ArgsBase:
    """fastmcp FuncMetadata 需要的逐层 dump 方法（见 ahedd.mcp.server）。"""

    def model_dump_one_level(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in type(self).model_fields}  # type: ignore[attr-defined]


def schema_to_args_model(tool_name: str, schema: dict[str, Any], *, bases: tuple[type, ...] | None = None) -> type:
    """JSON Schema -> pydantic args 模型（LangChain StructuredTool / MCP FuncMetadata 共用）。

    :param bases: 基类元组；缺省 (BaseModel, _ArgsBase)。MCP 侧需传 fastmcp 的 ArgModelBase。
    """
    from pydantic import BaseModel, Field, create_model

    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, field_def in properties.items():
        py_type = _JSON_TYPES.get(field_def.get("type", "string"), str)
        if field_name in required:
            fields[field_name] = (py_type, Field(..., description=field_def.get("description", "")))
        else:
            fields[field_name] = (py_type | None, Field(None, description=field_def.get("description", "")))
    base_classes = bases if bases is not None else (BaseModel, _ArgsBase)
    return create_model(f"{tool_name}_args", __base__=base_classes, **fields)


class ToolRegistry:
    """工具集的只读视图：按名查找、迭代、OpenAI 格式转换。"""

    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = list(tools)
        self._by_name = {t.name: t for t in self._tools}
        if len(self._by_name) != len(self._tools):
            raise ValueError("duplicate tool names in registry")

    @classmethod
    def from_list(cls, tools: list[ToolDefinition]) -> ToolRegistry:
        return cls(tools)

    def get(self, name: str) -> ToolDefinition | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return list(self._by_name)

    def __iter__(self):
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """转换为 OpenAI chat.completions 的 tools 参数格式。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools
        ]
