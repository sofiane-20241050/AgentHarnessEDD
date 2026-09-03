"""轨迹动力学指标：不依赖 LLM 判分的确定性过程度量。

覆盖维度（对齐调研报告 §6.4 与业界 Harness 评估维度）：
  - 轨迹动力学：有效动作比、冗余调用、工具错误/幻觉率、步数（Step-to-Success）
  - 自愈韧性：错误恢复（出现错误仍正常收卷）、失败连击
  - 资源开销：token 用量（帕累托分析的原料）
"""

from __future__ import annotations

from pydantic import BaseModel

from ahedd.trace.schema import Trajectory


class TrajectoryMetrics(BaseModel):
    """单条轨迹的过程指标。"""

    total_steps: int = 0
    turns: int = 0                          # 对话回合数（用户消息数）
    llm_rounds: int = 0                     # LLM 补全次数（assistant 事件数）
    tool_calls: int = 0
    distinct_tool_calls: int = 0            # 去重 (name, args) 后的调用数
    repeated_calls: int = 0                 # 冗余（重复相同调用）次数
    failed_calls: int = 0                   # 返回 ok=False / error 的调用
    error_events: int = 0                   # error 事件总数
    agent_errors: int = 0                   # Agent 自身错误（幻觉/坏参数/超轮次/工具业务失败）
    infra_errors: int = 0                   # 基础设施错误（网络/限流/超时，不计 Agent 头上）
    unknown_tool_calls: int = 0             # 幻觉：调用了不存在的工具
    malformed_args_calls: int = 0           # 幻觉：参数与 schema 不匹配
    useful_action_ratio: float = 1.0        # 有效动作比 = 去重调用 / 总调用
    tool_error_rate: float = 0.0            # Agent 错误 / (tool_call + Agent 错误)
    max_failure_streak: int = 0             # 最长失败连击（自愈韧性信号；infra 不计入）
    had_error: bool = False
    had_agent_error: bool = False
    recovered: bool | None = None           # Agent 出错后仍 stop 收卷（无 Agent 错误时无意义）
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_chars: int = 0                # 思考链总字符（思考依赖度信号）


def compute_trajectory_metrics(trajectory: Trajectory) -> TrajectoryMetrics:
    steps = trajectory.steps
    m = TrajectoryMetrics(
        total_steps=len(steps),
        turns=sum(1 for s in steps if s.type == "user"),
        llm_rounds=sum(1 for s in steps if s.type == "assistant"),
        tokens_in=trajectory.meta.total_usage.input_tokens,
        tokens_out=trajectory.meta.total_usage.output_tokens,
        reasoning_chars=sum(len(s.reasoning or "") for s in steps if s.type == "assistant"),
    )

    seen: set[tuple[str, str]] = set()
    tool_calls = 0
    repeated = 0
    for s in steps:
        if s.type != "tool_call":
            continue
        tool_calls += 1
        key = (s.tool_name or "", repr(sorted((s.tool_args or {}).items())))
        if key in seen:
            repeated += 1
        else:
            seen.add(key)
    for s in steps:
        if s.type == "tool_result" and isinstance(s.tool_result, dict) and s.tool_result.get("ok") is False:
            m.failed_calls += 1

    errors = [s for s in steps if s.type == "error"]
    agent_errors = [e for e in errors if e.error_kind != "infra"]  # 无 kind 的旧轨迹视同 Agent 侧
    infra_errors = [e for e in errors if e.error_kind == "infra"]
    m.tool_calls = tool_calls
    m.distinct_tool_calls = len(seen)
    m.repeated_calls = repeated
    m.error_events = len(errors)
    m.agent_errors = len(agent_errors)
    m.infra_errors = len(infra_errors)
    m.unknown_tool_calls = sum(1 for e in errors if "unknown tool" in e.content)
    m.malformed_args_calls = sum(1 for e in errors if "bad arguments" in e.content or "TypeError" in e.content)
    if tool_calls:
        m.useful_action_ratio = round(len(seen) / tool_calls, 4)
    if tool_calls + len(agent_errors):
        m.tool_error_rate = round(len(agent_errors) / (tool_calls + len(agent_errors)), 4)

    streak = current = 0
    for s in steps:
        if s.type == "error":
            if s.error_kind == "infra":
                continue  # 基础设施错误不计入 Agent 的失败连击
            current += 1
        elif s.type == "tool_call":
            continue
        else:
            current = 0
        streak = max(streak, current)
    m.max_failure_streak = streak

    m.had_error = bool(errors)
    m.had_agent_error = bool(agent_errors)
    if m.had_agent_error:
        m.recovered = all(
            s.stop_reason in ("stop", None) for s in steps if s.type == "assistant" and s.stop_reason
        ) and (steps[-1].type == "assistant" if steps else False)
    return m


def summarize_suite(
    runs: list[tuple[str, str, bool | None, TrajectoryMetrics]],
) -> dict:
    """套件级汇总：Step-to-Success 分布、错误恢复率、帕累托原料（成功率 vs token）。

    :param runs: (adapter, case_id, passed, metrics) 列表；passed=None 表示未判分
    """
    by_adapter: dict[str, list[tuple[str, bool | None, TrajectoryMetrics]]] = {}
    for adapter, case_id, passed, metrics in runs:
        by_adapter.setdefault(adapter, []).append((case_id, passed, metrics))

    summary: dict = {"adapters": {}}
    for adapter, entries in by_adapter.items():
        scored = [(c, p, m) for c, p, m in entries if p is not None]
        passed_entries = [(c, m) for c, p, m in scored if p]
        # 错误恢复率只针对 Agent 自身错误；infra 错误单独统计（基础设施质量信号）
        with_agent_errors = [m for _, _, m in scored if m.had_agent_error]
        recovered = [m for m in with_agent_errors if m.recovered]
        infra_runs = sum(1 for _, _, m in scored if m.infra_errors > 0)
        sts = sorted(m.turns for _, m in passed_entries)
        summary["adapters"][adapter] = {
            "runs": len(entries),
            "scored": len(scored),
            "pass_rate": round(len(passed_entries) / len(scored), 4) if scored else None,
            # Step-to-Success：成功轨迹的回合数分布（规划剪枝能力）
            "sts_median_turns": sts[len(sts) // 2] if sts else None,
            "sts_min_turns": sts[0] if sts else None,
            "sts_max_turns": sts[-1] if sts else None,
            "avg_useful_action_ratio": _avg([m.useful_action_ratio for _, _, m in scored]),
            "avg_tool_error_rate": _avg([m.tool_error_rate for _, _, m in scored]),
            # 错误恢复率：Agent 出错的运行中最终正常收卷的比例（自愈韧性）
            "error_recovery_rate": (
                round(len(recovered) / len(with_agent_errors), 4) if with_agent_errors else None
            ),
            # 基础设施错误影响的运行数（网络/限流，不计 Agent 失败）
            "infra_error_runs": infra_runs,
            "total_tokens_in": sum(m.tokens_in for _, _, m in entries),
            "total_tokens_out": sum(m.tokens_out for _, _, m in entries),
            # 帕累托原料：成功率 vs 平均 token 消耗
            "pareto": {
                "pass_rate": round(len(passed_entries) / len(scored), 4) if scored else None,
                "avg_tokens_per_run": round(
                    (sum(m.tokens_in + m.tokens_out for _, _, m in entries) / len(entries)) if entries else 0, 1
                ),
            },
        }
    return summary


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
