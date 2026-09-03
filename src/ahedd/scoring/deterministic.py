"""通道二：确定性断言（零成本、零幻觉）。

消费统一轨迹，按 TaskCase.rules 校验：
  - forbidden_tools：越权/禁调工具
  - max_turns：轮次上限
  - max_identical_calls：同工具同参数连续重复（死循环特征）
  - 失败恢复观察：连续失败调用次数（供分诊参考，不计违例）
"""

from __future__ import annotations

from ahedd.datasets.base import TaskCase
from ahedd.trace.schema import StepRecord, Trajectory


def check_trajectory_rules(case: TaskCase, trajectory: Trajectory) -> list[str]:
    """返回违例列表；空列表 = 全部通过。"""
    violations: list[str] = []
    rules = case.rules
    steps = trajectory.steps

    called = [s for s in steps if s.type == "tool_call" and s.tool_name]
    if bad := [s.tool_name for s in called if s.tool_name in set(rules.forbidden_tools)]:
        violations.append(f"forbidden tool called: {sorted(set(bad))}")

    turn_count = sum(1 for s in steps if s.type in ("user", "assistant"))
    if turn_count > rules.max_turns:
        violations.append(f"max_turns exceeded: {turn_count} > {rules.max_turns}")

    # 死循环检测：同一 (tool, args) 连续出现
    streak_key, streak_len = None, 0
    for s in called:
        key = (s.tool_name, repr(sorted((s.tool_args or {}).items())))
        streak_len = streak_len + 1 if key == streak_key else 1
        streak_key = key
        if streak_len > rules.max_identical_calls:
            violations.append(f"loop suspected: {s.tool_name} x{streak_len} identical calls")
            break

    return violations


def count_repeated_failures(steps: list[StepRecord]) -> int:
    """连续失败尝试的最大连击数——失败恢复能力的观测信号。

    一次失败尝试在轨迹上表现为 tool_call -> error 成对出现；
    重试的 tool_call 不打断连击，对话类事件（assistant/user/tool_result）归零。
    """
    longest = current = 0
    for s in steps:
        if s.type == "error":
            if s.error_kind == "infra":
                continue  # 基础设施错误（网络/限流）不计入 Agent 的失败连击
            current += 1
        elif s.type == "tool_call":
            continue  # 属于下一次尝试的调用，不中断连击
        else:
            current = 0
        longest = max(longest, current)
    return longest
