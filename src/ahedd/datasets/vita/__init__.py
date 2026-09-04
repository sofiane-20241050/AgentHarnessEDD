"""vita 数据集插件：VitaBench（ICLR 2026，美团 LongCat，MIT）映射进框架抽象。

数据适配框架（调研报告 §2.9）：不 vendor 代码，运行时 import `vita` 包。

安装（vitabench 源码可从 github.com/meituan-longcat/vitabench 克隆）::

    git clone https://github.com/meituan-longcat/vitabench
    uv pip install -e ./vitabench          # 提供 vita 包与 data/ 任务数据

映射关系：
  - 域：delivery / instore / ota / cross_domain（工具全集注册，如 delivery=20 个工具）
  - 任务：instructions -> instruction；user_profile -> UserScenario；
    state_rubrics + overall_rubrics -> rubrics；expected_states -> TaskCase.extra
    （确定性写断言原料）
  - 环境：每个任务自带初始 DB（environment 字段，含任务专属商店子集）——
    以 env_seed=任务索引 编码，reset(seed) 时重建对应初始态
"""

from __future__ import annotations

import asyncio
from typing import Any

from ahedd.datasets import register_dataset
from ahedd.datasets.base import TaskCase, UserScenario
from ahedd.env.base import default_diff
from ahedd.env.tools import ToolDefinition


class _force_utf8_open:
    """临时把进程内 open() 的缺省编码固定为 UTF-8。

    vita 源码在 Windows 中文区（默认 GBK）下 open() 不带 encoding 会读崩 UTF-8 数据；
    本上下文管理器在"导入 vita / 读任务文件 / 执行工具"期间兜底，不修改第三方代码。
    """

    def __enter__(self) -> _force_utf8_open:  # noqa: PYI034 - 返回类型即 self，ruff 误报于前向引用
        import builtins

        self._original = builtins.open

        def open_utf8(file, mode="r", *args, **kwargs):
            if "r" in mode and "b" not in mode and "encoding" not in kwargs:
                kwargs["encoding"] = "utf-8"
            return self._original(file, mode, *args, **kwargs)

        builtins.open = open_utf8
        return self

    def __exit__(self, *exc: object) -> None:
        import builtins

        builtins.open = self._original


try:
    with _force_utf8_open():
        from vita.domains.delivery.environment import get_environment as _get_delivery_env
        from vita.domains.delivery.environment import get_tasks as _get_delivery_tasks

    _VITA_AVAILABLE = True
except ImportError:  # 未安装 vita 包时插件静默不注册（datasets 加载器按 ImportError 跳过）
    _VITA_AVAILABLE = False

_LANGUAGES = {"zh": None, "en": "en"}
_DOMAINS = ("delivery", "instore", "ota", "cross_domain")


class VitaEnvironment:
    """包装 vita 的 Environment：工具全集注册；reset(seed=i) 重建第 i 个任务的初始 DB。"""

    def __init__(self, domain: str, language: str = "zh") -> None:
        self.domain = domain
        self.language = language
        self._tasks: list[Any] = _load_vita_tasks(domain, language)
        self._env: Any = None

    # ---- Environment 契约 ----

    def tools(self) -> list[ToolDefinition]:
        self._ensure_env()
        defs = []
        for tool in self._env.get_tools():
            defs.append(_wrap_tool(self._env, tool))
        return defs

    async def reset(self, seed: int | None = None) -> None:
        """seed = 任务索引：重建该任务的初始 DB（工具随 DB 重建）。"""
        index = int(seed or 0)
        if not 0 <= index < len(self._tasks):
            raise ValueError(f"env_seed={seed} 超出 {self.domain} 任务范围 [0, {len(self._tasks)})")
        db = _task_db(self._tasks[index])
        self._env = _build_env(self.domain, db, self.language)
        self._active_index = index

    def snapshot(self) -> dict[str, Any]:
        self._ensure_env()
        return self._env.tools.db.model_dump()

    def diff(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return default_diff(before, after)

    @property
    def active_task(self) -> Any:
        return self._tasks[self._active_index]

    def _ensure_env(self) -> None:
        if self._env is None:
            raise RuntimeError("环境未 reset：请先调用 reset(seed=任务索引)")


def _wrap_tool(vita_env: Any, tool: Any) -> ToolDefinition:
    schema = tool.params.model_json_schema() if tool.params else {"type": "object", "properties": {}}
    schema.pop("title", None)

    async def call(**kwargs: Any) -> Any:
        def _run() -> Any:
            with _force_utf8_open():
                return vita_env.use_tool(tool_name=tool.name, **kwargs)

        return await asyncio.to_thread(_run)

    return ToolDefinition(
        name=tool.name,
        description=(tool.long_desc or tool.short_desc or tool.name)[:1024],
        parameters=schema,
        func=call,
    )


def _load_vita_tasks(domain: str, language: str) -> list[Any]:
    with _force_utf8_open():  # vita 以默认编码读任务 JSON，Windows 中文区需兜底 UTF-8
        return _load_vita_tasks_inner(domain, language)


def _load_vita_tasks_inner(domain: str, language: str) -> list[Any]:
    lang = _LANGUAGES.get(language, None)
    if domain == "delivery":
        return _get_delivery_tasks(language=lang)
    from vita.domains.instore.environment import get_tasks as get_instore_tasks
    from vita.domains.ota.environment import get_tasks as get_ota_tasks

    if domain == "instore":
        return get_instore_tasks(language=lang)
    if domain == "ota":
        return get_ota_tasks(language=lang)
    if domain == "cross_domain":
        # 跨场景域：合并三域任务（域标记保留在 task.domain）
        merged: list[Any] = []
        merged.extend(_get_delivery_tasks(language=lang))
        merged.extend(get_instore_tasks(language=lang))
        merged.extend(get_ota_tasks(language=lang))
        return merged
    raise ValueError(f"vita 未知域: {domain!r}")


def _build_env(domain: str, db: dict[str, Any], language: str) -> Any:
    lang = _LANGUAGES.get(language, None)
    if domain == "delivery":
        return _get_delivery_env(db=db, language=lang)
    from vita.domains.instore.environment import get_environment as get_instore_env
    from vita.domains.ota.environment import get_environment as get_ota_env

    if domain == "instore":
        return get_instore_env(db=db, language=lang)
    if domain == "ota":
        return get_ota_env(db=db, language=lang)
    if domain == "cross_domain":
        # 跨场景：按任务自身域路由构建（VitaBench 通过去策略框架合并域工具）
        raise NotImplementedError("cross_domain 的环境构建按任务域路由，见 examples/vitabench_mcp")
    raise ValueError(domain)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value) if isinstance(value, dict) else {}


def _task_db(task: Any) -> dict[str, Any]:
    env = task.environment
    if hasattr(env, "model_dump"):
        return env.model_dump()
    return dict(env)


class VitaProvider:
    name = "vita"

    def __init__(self, language: str = "zh") -> None:
        self.language = language
        self._task_cache: dict[str, dict[str, Any]] = {}  # case_id -> task（组装任务级 system prompt 用）

    def domains(self) -> list[str]:
        return list(_DOMAINS)

    def agent_system_prompt(self, case: TaskCase, *, solo: bool = False) -> str:
        """VitaBench 官方 Agent 系统提示：模板.format(time=任务时间+星期)。

        多轮（用户模拟器）用 agent_system_prompt（含对话规范与 ###STOP### 约定），
        单发直跑用 solo_agent_system_prompt（一次性完成、不与用户交互）。
        """
        from vita.prompts import get_prompts
        from vita.utils.utils import get_weekday

        task = self._task_cache.get(case.id)
        time_str = str(_as_dict(task).get("environment", {}).get("time") or "") if task else ""
        if task is not None:
            time_str = str(_task_db(task).get("time") or "")
        lang = _LANGUAGES.get(self.language, None)
        prompts = get_prompts()
        template = prompts.solo_agent_system_prompt if solo else prompts.agent_system_prompt
        text = template if isinstance(template, str) else str(getattr(template, "chinese", "") or template)
        if lang == "en":
            text = str(getattr(template, "english", "") or text)
        if time_str:
            try:
                return text.format(time=f"{time_str} {get_weekday(time_str, self.language)}")
            except Exception:  # noqa: BLE001 - 模板占位符不匹配时退回原文
                return text
        return text

    def load(self, domain: str) -> list[TaskCase]:
        cases = []
        for index, task in enumerate(_load_vita_tasks(domain, self.language)):
            self._task_cache[str(task.id)] = task
            ec = _as_dict(task.evaluation_criteria)
            rubrics: list[str] = []
            for state in ec.get("expected_states", []):
                rubrics.extend(state.get("state_rubrics", []))
            rubrics.extend(ec.get("overall_rubrics", []))
            profile = _as_dict(_as_dict(task.user_scenario).get("user_profile"))
            cases.append(
                TaskCase(
                    id=str(task.id),
                    domain=domain,
                    instruction=str(task.instructions or ""),
                    user_scenario=UserScenario(
                        persona=str(profile.get("性格") or profile.get("personality") or "") or None,
                        traits={k: str(v) for k, v in profile.items()},
                    ),
                    rubrics=rubrics,
                    env_seed=index,  # 任务索引 -> 环境初始态
                    source="vitabench",
                    extra={"evaluation_criteria": ec} if ec else {},
                )
            )
        return cases

    def build_environment(self, domain: str) -> VitaEnvironment:
        return VitaEnvironment(domain, language=self.language)


if _VITA_AVAILABLE:
    register_dataset("vita")(
        lambda: VitaProvider()
    )
