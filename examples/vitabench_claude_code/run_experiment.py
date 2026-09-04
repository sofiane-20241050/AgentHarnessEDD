#!/usr/bin/env python3
"""端到端实战脚本：Claude Code CLI × VitaBench（MCP 原生工具 + 用户模拟器 + 判分 + 报告）。

用法（在本目录下）::

    python run_experiment.py --cases 10711002
    python run_experiment.py --cases 10711002,10711003 --skip-report

前置条件见同目录 README.md（模型端点 / claude CLI / vitabench 安装 / .env 配置）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 允许直接从 examples 目录运行：把仓库根加进 sys.path（框架以 -e 安装时可省）
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ahedd.adapters.claude_code_adapter import ClaudeCodeAdapter  # noqa: E402
from ahedd.config import load_dotenv, load_models_config  # noqa: E402
from ahedd.datasets import get_dataset  # noqa: E402
from ahedd.llm import make_client  # noqa: E402
from ahedd.report.html import render_report  # noqa: E402
from ahedd.runner import run_dataset  # noqa: E402
from ahedd.scoring import RubricSlidingWindowScorer, compute_trajectory_metrics  # noqa: E402
from ahedd.user import UserSimulator  # noqa: E402

MCP_PORT = 8023


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude Code CLI × VitaBench 端到端实验")
    parser.add_argument("--cases", required=True, help="逗号分隔的 case id，如 10711002")
    parser.add_argument("--domain", default="delivery", help="vita 域（默认 delivery）")
    parser.add_argument("--mcp-port", type=int, default=MCP_PORT)
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    roles = load_models_config()
    print(f"[1/4] 被测模型: {roles.agent.model} | 判分器: {roles.judge.model if roles.judge else '(未配)'}")

    provider = get_dataset("vita")
    cases = [c for c in provider.load(args.domain) if c.id in set(args.cases.split(","))]
    print(f"[1/4] vita/{args.domain}: 命中 {len(cases)} 个任务 {[c.id for c in cases]}")

    # ② 驱动 Claude Code（MCP 原生工具模式；环境 MCP server 由 CLI/外部负责拉起，
    #    本脚本聚焦流程演示，也可改用 `ahedd run --tool-mode mcp` 全自动编排）
    def adapter_factory() -> ClaudeCodeAdapter:
        return ClaudeCodeAdapter(
            workdir=os.environ.get("AHEDD_CC_DIR", "."),
            ssh_target=os.environ.get("AHEDD_CC_SSH") or None,
            node_bin=os.environ.get("AHEDD_CC_NODE_BIN", "~/.nvm/versions/node/v22.22.0/bin"),
            tool_mode="mcp",
            mcp_url=f"http://127.0.0.1:{args.mcp_port}/mcp",
            events_file=os.environ.get("AHEDD_CC_EVENTS", "/tmp/ahedd_mcp_events.jsonl"),
            events_ssh_target=os.environ.get("AHEDD_CC_SSH") or None,
        )

    def sim_factory(case):  # noqa: ANN001, ANN202
        if roles.user_simulator is None:
            return None
        return UserSimulator(
            make_client(roles.user_simulator), case,
            chat_kwargs={"max_tokens": 512, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )

    print("[2/4] 跑评测（MCP 原生工具 + 用户模拟器多轮）… 先确保 MCP server 在跑：")
    print(f"        ahedd mcp serve --dataset vita --domain {args.domain} --port {args.mcp_port} "
          f"--events-file ${{AHEDD_CC_EVENTS:-/tmp/ahedd_mcp_events.jsonl}}")
    pairs = await run_dataset(
        provider=provider,
        adapter_factory=adapter_factory,
        domain=args.domain,
        case_ids=args.cases.split(","),
        dataset="vita",
        agent_model=roles.agent.model,
        user_simulator_factory=sim_factory if roles.user_simulator else None,
    )

    # ③ 判分 + 过程指标
    print("[3/4] 离线判分…")
    judge = None
    if roles.judge is not None:
        judge = RubricSlidingWindowScorer(
            make_client(roles.judge),
            chat_kwargs={"max_tokens": 2048, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
    rows = []
    for case, outcome in pairs:
        passed = None
        rubric_n = (0, 0)
        if judge is not None:
            report = await judge.score(case, outcome.trajectory)
            passed = report.passed
            rubric_n = (sum(r.satisfied for r in report.rubric_results), len(report.rubric_results))
            (Path("runs") / outcome.trajectory.meta.domain / case.id / f"{outcome.trajectory.meta.run_id}.score.json").write_text(
                report.model_dump_json(indent=1), encoding="utf-8"
            )
        metrics = compute_trajectory_metrics(outcome.trajectory)
        status = "----" if passed is None else ("PASS" if passed else "FAIL")
        print(f"    [{status}] {case.id} rubric={rubric_n[0]}/{rubric_n[1]} "
              f"turns={metrics.turns} tokens={metrics.tokens_in}/{metrics.tokens_out}")
        rows.append((case, outcome, passed))

    # ④ 报告
    if not args.skip_report:
        print("[4/4] 渲染 HTML 报告…")
        items = [{
            "meta": o.trajectory.meta.model_dump(),
            "steps": [s.model_dump(exclude_none=True) for s in o.trajectory.steps],
            "score": None, "env_diff": o.env_diff,
        } for _, o, _ in rows]
        out = render_report(items, "vita_report.html")
        print(f"    report -> {out}")

    print("\n完成。冻结失败轨迹：ahedd freeze <run_id> --attribution <归因标签>")


if __name__ == "__main__":
    asyncio.run(main())
