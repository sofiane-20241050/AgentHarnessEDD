"""通道一：rubric 滑动窗口 LLM 判分器（D3 实现）。

设计依据 VitaBench 论文 §3.3（人工一致性 κ=0.828 的配置），借鉴 Prometheus 2 的
rubric 结构与 G-Eval 的 CoT 判分（见调研报告 §9）：

  1. 任务 rubric 为人工编写的原子断言清单（TaskCase.rubrics，完全自定义）
  2. 轨迹按"回合"切重叠窗口（默认 w=10、重叠 δ=2，可配置）
  3. 逐窗口调用 judge：输入 = 窗口转录 + rubric 当前状态向量；输出 = 更新的状态
  4. 状态粘滞（sticky）：某条断言一旦满足即永久标记（证据可能出现在早期）
  5. 终判全有或全无 score = 1[全部满足]；rubric 级明细保留用于分环节诊断

判分器是轨迹的离线消费者（先存轨迹、后判分），可换模型、可重跑。
"""

from __future__ import annotations

import json
import re
from typing import Any

from ahedd.datasets.base import TaskCase
from ahedd.llm.base import LLMClient
from ahedd.scoring.base import RubricResult, ScoreReport
from ahedd.scoring.deterministic import check_trajectory_rules
from ahedd.trace.schema import StepRecord, Trajectory

_JUDGE_SYSTEM = """你是 Agent 评测判分器。给你一段 Agent 交互轨迹的窗口，以及各评分项(rubric)的当前状态。
判断每个评分项在本窗口内是否被满足。规则：
- 已标记 satisfied=true 的项保持 true（永久满足，不因后续窗口撤销）
- 只依据窗口内可观察的证据判定（助手发言、工具调用名称与参数、工具返回）
- 工具返回是环境的真实反馈，优先于助手的口头声称
- evidence_turn 填该断言被满足/违反时所在的回合号（本窗口内），无法定位填 null
输出严格 JSON，不要输出任何其他文字。"""

_JUDGE_USER_TMPL = """<rubrics>
{rubrics_json}
</rubrics>
<window>
第 {window_index} 个窗口（回合 {turn_start}-{turn_end}，与上一窗口重叠 {overlap} 回合）：
{transcript}
</window>

请输出 JSON：
{{"results": [{{"key": "r0", "satisfied": true, "evidence_turn": 3, "rationale": "一句话依据"}}, ...]}}
每个 rubric 一项，key 与输入一致。"""


class RubricSlidingWindowScorer:
    """rubric 滑窗判分器。judge 为任意 LLMClient（OpenAI 兼容端点 / Fake）。"""

    def __init__(
        self,
        judge_client: LLMClient,
        *,
        window_turns: int = 10,
        overlap_turns: int = 2,
        chat_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.judge = judge_client
        self.window_turns = window_turns
        self.overlap_turns = overlap_turns
        # judge 需要 JSON 输出；思考型判分模型建议传入关闭思考的 extra_body
        self.chat_kwargs = chat_kwargs or {}

    async def score(self, case: TaskCase, trajectory: Trajectory) -> ScoreReport:
        if not case.rubrics:
            raise ValueError(f"case {case.id!r} 没有定义 rubrics，无法判分")
        states: dict[str, dict[str, Any]] = {
            f"r{i}": {"description": desc, "satisfied": False, "evidence_turn": None, "rationale": ""}
            for i, desc in enumerate(case.rubrics)
        }
        spec = getattr(self.judge, "spec", None)
        judge_meta: dict[str, str] = {
            "judge": getattr(spec, "model", None) or type(self.judge).__name__,
            "window_turns": str(self.window_turns),
            "overlap_turns": str(self.overlap_turns),
        }

        for window_index, (steps, turn_start, turn_end) in enumerate(self._windows(trajectory.steps)):
            prompt = self._build_prompt(states, steps, window_index, turn_start, turn_end)
            resp = await self.judge.chat(
                [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                **self.chat_kwargs,
            )
            for item in _parse_results(resp.content or ""):
                state = states.get(str(item.get("key")))
                if state is None:
                    continue
                if state["satisfied"]:
                    continue  # 粘滞：已满足不撤销
                if bool(item.get("satisfied")):
                    state["satisfied"] = True
                    state["evidence_turn"] = item.get("evidence_turn")
                    state["rationale"] = str(item.get("rationale", ""))[:300]

        rubric_results = [
            RubricResult(
                key=key,
                description=state["description"],
                satisfied=state["satisfied"],
                evidence_turn=state["evidence_turn"],
            )
            for key, state in states.items()
        ]
        satisfied_count = sum(r.satisfied for r in rubric_results)
        violations = check_trajectory_rules(case, trajectory)
        passed = satisfied_count == len(rubric_results) and not violations
        return ScoreReport(
            task_id=case.id,
            passed=passed,
            score=1.0 if satisfied_count == len(rubric_results) else satisfied_count / len(rubric_results),
            rubric_results=rubric_results,
            rule_violations=violations,
            judge_meta=judge_meta,
        )

    # ---- 窗口切分与转录 ----

    def _windows(self, steps: list[StepRecord]) -> list[tuple[list[StepRecord], int, int]]:
        """按回合（user/assistant 事件）切重叠窗口；工具事件归属其前面的回合。"""
        if not steps:
            return []
        turn_bounds: list[tuple[int, int]] = []  # (start_step_idx, end_step_idx)
        current_start: int | None = None
        for idx, step in enumerate(steps):
            if step.type in ("user", "assistant"):
                if current_start is not None:
                    turn_bounds.append((current_start, idx - 1))
                current_start = idx
        if current_start is not None:
            turn_bounds.append((current_start, len(steps) - 1))

        windows: list[tuple[list[StepRecord], int, int]] = []
        w, delta = self.window_turns, self.overlap_turns
        n = len(turn_bounds)
        start_turn = 0
        while start_turn < n:
            end_turn = min(start_turn + w, n) - 1
            step_from = turn_bounds[start_turn][0]
            step_to = turn_bounds[end_turn][1]
            windows.append((steps[step_from : step_to + 1], start_turn + 1, end_turn + 1))
            if end_turn >= n - 1:
                break
            # 精确共享 delta 个回合（对齐论文语义）：
            # [1..4] w=4 δ=1 -> 下一窗口 [4..7]，重叠 {4} 恰为 1 个回合；
            # max 保底：退化配置（delta >= w）时退化为不重叠逐格滑动
            start_turn = max(end_turn - delta + 1, start_turn + 1)
        return windows

    def _build_prompt(
        self,
        states: dict[str, dict[str, Any]],
        steps: list[StepRecord],
        window_index: int,
        turn_start: int,
        turn_end: int,
    ) -> str:
        rubrics_json = json.dumps(
            [
                {"key": key, "description": s["description"], "current_satisfied": s["satisfied"]}
                for key, s in states.items()
            ],
            ensure_ascii=False,
            indent=1,
        )
        return _JUDGE_USER_TMPL.format(
            rubrics_json=rubrics_json,
            window_index=window_index,
            turn_start=turn_start,
            turn_end=turn_end,
            overlap=self.overlap_turns,
            transcript=_render_transcript(steps),
        )


def _render_transcript(steps: list[StepRecord], max_chars: int = 6000) -> str:
    """窗口内容转录：回合号 + 事件摘要（截断防爆窗口）。"""
    lines: list[str] = []
    turn = 0
    for step in steps:
        if step.type == "user":
            turn += 1
            lines.append(f"[回合{turn}] 用户: {step.content}")
        elif step.type == "assistant":
            if step.content.strip():
                lines.append(f"[回合{turn}] 助手: {step.content}")
        elif step.type == "tool_call":
            args = json.dumps(step.tool_args or {}, ensure_ascii=False)
            lines.append(f"[回合{turn}] 工具调用 {step.tool_name}({args})")
        elif step.type == "tool_result":
            result = json.dumps(step.tool_result, ensure_ascii=False, default=str)
            lines.append(f"[回合{turn}] 工具返回 {step.tool_name} -> {result}")
        elif step.type == "error":
            lines.append(f"[回合{turn}] 错误: {step.content}")
        else:  # plan/subagent/memory 等 harness 事件
            label = {"plan": "规划", "subagent": "子Agent", "memory": "记忆/文件"}.get(step.type, step.type)
            args = json.dumps(step.tool_args or {}, ensure_ascii=False)
            lines.append(f"[回合{turn}] {label}: {step.tool_name} {args}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...（截断，完整窗口 {len(text)} 字符）"
    return text


def _parse_results(text: str) -> list[dict[str, Any]]:
    """鲁棒解析 judge 输出的 JSON（容忍 ```json 围栏与 <think> 噪声）。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return []
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    results = data.get("results", data if isinstance(data, list) else [])
    return [r for r in results if isinstance(r, dict)]
