"""单任务评测编排：环境重置 -> 工具包装录制 -> 驱动被测 Agent -> 落盘轨迹。

原则：先存轨迹、后判分（判分器可换、可离线重跑，见调研报告 §2.8）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ahedd.adapters.base import AgentAdapter, TaskInput
from ahedd.datasets.base import DatasetProvider, TaskCase
from ahedd.env.base import Environment
from ahedd.trace.schema import RunMeta, Trajectory, TrajectoryRecorder, Usage


@dataclass
class CaseOutcome:
    """单任务单次运行的结果（未判分）。"""

    trajectory: Trajectory
    final_message: str
    stop_reason: str
    env_diff: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


async def run_dataset(
    *,
    provider: DatasetProvider,
    adapter_factory: Callable[[], AgentAdapter],
    domain: str | None = None,
    case_ids: list[str] | None = None,
    trials: int = 1,
    trace_dir: str = "runs",
    dataset: str = "",
    agent_model: str = "",
    concurrency: int = 1,
    on_result: Callable[[TaskCase, CaseOutcome, int], None] | None = None,
    case_setup: Callable[[TaskCase], Any] | None = None,
    user_simulator_factory: Callable[[TaskCase], Any] | None = None,
) -> list[tuple[TaskCase, CaseOutcome]]:
    """批量跑一个数据集域：每任务独立环境、独立适配器实例；trials 次采样（Pass^k 基础）。

    :param concurrency: 并发上限（信号量）。环境互不共享、轨迹各写各文件，天然隔离。
    :param on_result: 每完成一条立即回调 (case, outcome, trial)——进度展示用。
    :param case_setup: 每个任务开始前调用（同步或 async）——如 MCP server 按 case 重启；
        与并发互斥（per-case 全局资源只能串行）。
    :param user_simulator_factory: 任务 -> UserSimulator 工厂（双 LLM 交互数据集用，
        目前 openai-loop 车道支持多轮对话模式）。
    """
    if case_setup is not None and concurrency > 1:
        raise ValueError("case_setup（按 case 重启等）与 concurrency>1 互斥")
    domain = domain or provider.domains()[0]
    cases = provider.load(domain)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.id in wanted]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(case: TaskCase, trial: int) -> tuple[TaskCase, CaseOutcome]:
        async with semaphore:
            if case_setup is not None and trial == 1:
                maybe_await = case_setup(case)
                if maybe_await is not None and hasattr(maybe_await, "__await__"):
                    await maybe_await
            env = provider.build_environment(domain)
            # 数据集自带的 Agent 系统提示（如 VitaBench 官方模板，任务级时间注入）：
            # 多轮（有用户模拟器）用交互版模板，单发用 solo 版
            system_prompt = None
            sp_fn = getattr(provider, "agent_system_prompt", None)
            if callable(sp_fn):
                try:
                    system_prompt = sp_fn(case, solo=user_simulator_factory is None)
                except Exception:  # noqa: BLE001 - 提示组装失败退回适配器默认
                    system_prompt = None
            outcome = await run_case(
                dataset=dataset or provider.name,
                adapter=adapter_factory(),
                env=env,
                case=case,
                task_id=case.id,
                instruction=case.instruction,
                system_prompt=system_prompt,
                env_seed=case.env_seed,
                trace_dir=trace_dir,
                agent_model=agent_model,
                user_simulator_factory=user_simulator_factory,
            )
            if on_result:
                on_result(case, outcome, trial)
            return case, outcome

    tasks = [asyncio.create_task(_run_one(case, trial + 1)) for case in cases for trial in range(trials)]
    pairs = await asyncio.gather(*tasks)
    return sorted(pairs, key=lambda p: (p[0].id, p[1].trajectory.meta.run_id))


async def run_case(
    *,
    dataset: str,
    adapter: AgentAdapter,
    env: Environment,
    task_id: str,
    instruction: str,
    system_prompt: str | None = None,
    env_seed: int | None = None,
    trace_dir: str = "runs",
    agent_model: str = "",
    case: TaskCase | None = None,
    user_simulator_factory: Callable[[TaskCase], Any] | None = None,
) -> CaseOutcome:
    """评测单任务一次。

    多轮交互：提供 user_simulator_factory 时（数据集任务携带剧本/画像），
    用户模拟器产出开场白并逐轮回应 Agent 的终答（###STOP### 结束）；
    模拟器用量并入 total_usage。未提供时为单轮指令直跑。
    """
    meta = RunMeta(
        task_id=task_id, domain=env.domain, dataset=dataset, adapter=adapter.name,
        agent_model=agent_model, system_prompt=system_prompt,
    )
    recorder = TrajectoryRecorder(meta)

    await env.reset(env_seed)
    before = env.snapshot()

    simulator = None
    if user_simulator_factory is not None and case is not None:
        simulator = user_simulator_factory(case)
        opening = await simulator.start()
        instruction = opening  # Agent 的首轮输入是模拟器说出的"人话"，而非原始剧本
    recorder.note("user", content=instruction)
    tools = [recorder.wrap_tool(t) for t in env.tools()]
    task = TaskInput(task_id=task_id, instruction=instruction, system_prompt=system_prompt)

    adapter_final_state: dict[str, Any] | None = None
    try:
        run_kwargs = {}
        if simulator is not None:
            run_kwargs["user_responder"] = simulator.reply
        result = await adapter.run(task, tools, recorder, **run_kwargs)
        u = result.usage_total or {}
        total_in = int(u.get("input_tokens", 0) or 0)
        total_out = int(u.get("output_tokens", 0) or 0)
        cost = float(u.get("cost_usd", 0.0) or 0.0)
        if simulator is not None:  # 用户模拟器用量并入任务总账
            total_in += simulator.total_usage.input_tokens
            total_out += simulator.total_usage.output_tokens
            cost += simulator.total_usage.cost_usd
        recorder.trajectory.meta.total_usage = Usage(
            input_tokens=total_in, output_tokens=total_out, cost_usd=cost
        )
        adapter_final_state = result.final_state  # MCP 等外置执行车道的终态快照
        outcome = CaseOutcome(
            trajectory=recorder.trajectory,
            final_message=result.final_message,
            stop_reason=result.stop_reason,
            env_diff=result.env_diff,  # None 则下方本地计算
        )
    except Exception as exc:  # noqa: BLE001 - 失败也是评测结果，必须入轨
        from ahedd.trace.errors import classify_exception

        last = recorder.trajectory.steps[-1] if recorder.trajectory.steps else None
        already_noted = (
            last is not None and last.type == "error" and last.content.startswith(type(exc).__name__)
        )
        if not already_noted:  # 适配器已入轨的异常（如 infra）不重复记
            recorder.note(
                "error",
                content=f"{type(exc).__name__}: {exc}",
                error_kind=classify_exception(exc),
            )
        outcome = CaseOutcome(
            trajectory=recorder.trajectory,
            final_message="",
            stop_reason="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    if outcome.env_diff is None:
        outcome.env_diff = env.diff(before, env.snapshot())
    final_state = adapter_final_state or env.snapshot()
    path = f"{trace_dir}/{env.domain}/{task_id}/{meta.run_id}.jsonl"
    recorder.dump_jsonl(path)
    # 旁车持久化：终态 diff（报告消费） + 终态全量快照（确定性终态断言消费）
    import json as _json
    from pathlib import Path as _P

    _P(path).with_suffix(".envdiff.json").write_text(
        _json.dumps(outcome.env_diff, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _P(path).with_suffix(".envstate.json").write_text(
        _json.dumps(final_state, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return outcome
