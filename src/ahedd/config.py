"""LLM 端点配置：分层解析。

优先级（后者覆盖前者）::

    内置默认值  <  models.yaml  <  .env / 环境变量  <  CLI 显式传入（overrides）

环境变量命名 ``AHEDD_{ROLE}_{FIELD}``：

- ROLE   ∈ AGENT / USER_SIMULATOR / JUDGE
- FIELD  ∈ NAME / MODEL / PROVIDER / BASE_URL / API_KEY / API_KEY_ENV / MAX_TOKENS / TEMPERATURE

``.env``（默认读取仓库根）在解析时注入环境变量（不覆盖已存在的真实环境变量）。
密钥解析：``api_key`` 显式值 > ``api_key_env`` 指向的环境变量 > "EMPTY"（vLLM 本地端点）。
配置文件与代码中都不落真实密钥。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ENV_PREFIX = "AHEDD_"
ROLES = ("agent", "user_simulator", "judge")

_FIELD_TYPES: dict[str, type] = {
    "name": str,
    "model": str,
    "provider": str,
    "base_url": str,
    "api_key": str,
    "api_key_env": str,
    "max_tokens": int,
    "temperature": float,
}


class ModelSpec(BaseModel):
    """单个 LLM 端点。OpenAI 兼容：base_url + model 即可指向 vLLM / OpenRouter / 官方 API。

    provider 为预留接缝：当前仅 "openai"（OpenAI SDK 兼容实现）与测试用 "fake"；
    未来接其他 SDK 时在 ahedd.llm.make_client 工厂分发，上层零改动。
    """

    name: str
    model: str
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int | None = None
    temperature: float = 0.0
    extra_headers: dict[str, str] = Field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """密钥解析：显式 api_key > api_key_env 环境变量 > "EMPTY"。"""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env, "EMPTY")


class LLMRoles(BaseModel):
    """一次评测的三角色端点集合。agent 必填；user_simulator / judge 可后补。"""

    agent: ModelSpec
    user_simulator: ModelSpec | None = None
    judge: ModelSpec | None = None


# ---- 分层加载 ----


def _parse_dotenv(text: str) -> dict[str, str]:
    """解析 KEY=VALUE（忽略空行与 # 注释，去除引号包裹）。"""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        value = raw.strip().strip('"').strip("'")
        if key.strip():
            values[key.strip()] = value
    return values


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """读取 .env 并注入环境变量（setdefault：真实环境变量优先）。返回解析结果。"""
    p = Path(path)
    if not p.exists():
        return {}
    values = _parse_dotenv(p.read_text(encoding="utf-8"))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def _coerce(field: str, value: str) -> Any:
    target = _FIELD_TYPES.get(field, str)
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    return value


def load_models_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
    dotenv_path: str | Path | None = ".env",
    env: Mapping[str, str] | None = None,
) -> LLMRoles:
    """分层合并并构建 LLMRoles。

    :param path: models.yaml 路径，默认 "models.yaml"，不存在则跳过该层
    :param overrides: CLI 显式层，形如 ``{"agent": {"model": "..."}}``
    :param dotenv_path: .env 路径，默认 ".env"，None 跳过
    :param env: 环境变量来源（默认 os.environ；测试可注入）
    """
    merged: dict[str, dict[str, Any]] = {role: {} for role in ROLES}

    # 层 1：YAML（缺省跳过）
    yaml_path = Path(path) if path is not None else Path("models.yaml")
    if yaml_path.exists():
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"invalid models config (expected mapping): {yaml_path}")
        for role in ROLES:
            section = raw.get(role)
            if isinstance(section, dict):
                merged[role].update(section)

    # 层 2：.env / 环境变量
    dotenv_values: dict[str, str] = {}
    if dotenv_path is not None:
        dotenv_values = load_dotenv(dotenv_path)
    effective_env: Mapping[str, str]
    if env is None:
        effective_env = os.environ  # load_dotenv 已 setdefault，真实环境变量仍优先
    else:
        effective_env = {**dotenv_values, **env}

    for role in ROLES:
        for field in _FIELD_TYPES:
            key = f"{ENV_PREFIX}{role.upper()}_{field.upper()}"
            if key in effective_env:
                merged[role][field] = _coerce(field, effective_env[key])

    # 层 3：CLI 显式传入
    for role, section in (overrides or {}).items():
        if role not in merged:
            raise ValueError(f"unknown role in overrides: {role!r} (expected {ROLES})")
        merged[role].update(section)

    # 构建：非空角色才建 spec；name 缺省用角色名
    specs: dict[str, ModelSpec] = {}
    for role, fields in merged.items():
        if not fields:
            continue
        fields.setdefault("name", role)
        specs[role] = ModelSpec.model_validate(fields)

    agent = specs.get("agent")
    if agent is None:
        raise ValueError(
            "agent 模型未配置：请设置 AHEDD_AGENT_MODEL / models.yaml / CLI 参数之一"
        )
    return LLMRoles(
        agent=agent,
        user_simulator=specs.get("user_simulator"),
        judge=specs.get("judge"),
    )
