"""ahedd 命令行入口。

命令集（随里程碑逐步落地）：
  ahedd datasets list          已注册数据集
  ahedd adapters list          已注册被测适配器
  ahedd run --dataset --domain --adapter [--model ...]   跑评测
  ahedd report --run <run_id>                            生成 HTML 诊断报告
  ahedd freeze <run_id>       失败轨迹冻结为回归用例
  ahedd ci                    回归门禁（全量回归用例红绿检查）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click


@click.group()
@click.version_option(package_name="agentharness-edd")
def main() -> None:
    """AgentHarnessEDD —— 面向 Agent Harness 的评估驱动开发（EDD）框架。"""


@main.command("datasets")
def datasets_cmd() -> None:
    """列出已注册的评测数据集。"""
    from ahedd.datasets import list_datasets

    names = list_datasets() or ["（暂无：首个数据集 vita 于 D1 接入）"]
    click.echo("\n".join(names))


@main.command("adapters")
def adapters_cmd() -> None:
    """列出已注册的被测 Agent 适配器。"""
    from ahedd.adapters import list_adapters

    names = list_adapters() or ["（暂无）"]
    click.echo("\n".join(names))


@main.command("run")
@click.option("--dataset", default="mock", show_default=True, help="数据集名（见 datasets list）")
@click.option("--domain", default=None, help="域（缺省取数据集第一个域）")
@click.option("--cases", "case_ids", default=None, help="逗号分隔的 case id，缺省全量")
@click.option("--adapter", default="openai-loop", show_default=True, help="被测适配器（当前仅 openai-loop 已实现）")
@click.option("--models", "models_yaml", default=None, help="models.yaml 路径（缺省仅用 .env/环境变量层）")
@click.option("--trials", default=1, show_default=True, help="每任务独立运行次数 k（Pass^k 采样）")
@click.option("--concurrency", default=1, show_default=True, help="并发上限（信号量；环境按任务隔离，可安全并发）")
@click.option("--max-tokens", default=4096, show_default=True, help="单次补全最大 token")
@click.option("--disable-thinking", is_flag=True, help="vLLM: extra_body enable_thinking=False（Qwen3 系思考模型提速）")
@click.option(
    "--tool-mode", "tool_mode", type=click.Choice(["text", "mcp"]), default="text", show_default=True,
    help="工具注入：text=协议桥兜底；mcp=原生 MCP（当前支持 claude-code）",
)
@click.option("--mcp-port", default=8023, show_default=True, help="MCP server 端口（tool-mode=mcp 时）")
@click.option("--no-user-sim", is_flag=True, help="禁用用户模拟器（默认：配置了 AHEDD_USER_SIMULATOR_* 且任务带剧本时启用）")
@click.option("--cc-timeout", default=900, show_default=True, help="claude-code 车道：单次 claude -p 子进程超时（秒，含其内部工具循环）")
def run_cmd(
    dataset: str,
    domain: str | None,
    case_ids: str | None,
    adapter: str,
    models_yaml: str | None,
    trials: int,
    concurrency: int,
    max_tokens: int,
    disable_thinking: bool,
    tool_mode: str,
    mcp_port: int,
    no_user_sim: bool,
    cc_timeout: int,
) -> None:
    """跑评测：采集轨迹 + 确定性断言（rubric 判分用 ahedd score 离线执行，先存后判）。"""
    import asyncio
    import json as _json
    import os

    from ahedd.config import load_models_config
    from ahedd.datasets import get_dataset
    from ahedd.llm import make_client
    from ahedd.runner import run_dataset
    from ahedd.scoring.deterministic import check_trajectory_rules

    roles = load_models_config(models_yaml)
    # MCP URL：AHEDD_MCP_URL 完整覆盖（端口从 URL 解析），否则由 --mcp-port 构造
    import os as _os
    from urllib.parse import urlsplit

    _mcp_url = _os.environ.get("AHEDD_MCP_URL") or f"http://127.0.0.1:{mcp_port}/mcp"
    if _os.environ.get("AHEDD_MCP_URL"):
        mcp_port = urlsplit(_mcp_url).port or mcp_port
    click.echo(
        f"# dataset={dataset} adapter={adapter} model={roles.agent.model} "
        f"trials={trials} concurrency={concurrency}"
    )

    # --max-tokens 对所有适配器生效：统一写进 agent spec
    agent_spec = roles.agent.model_copy(update={"max_tokens": max_tokens})

    def _cc_live_print(event: dict) -> None:
        """claude-code 实时事件（stream-json）：逐 turn 可见工具调用与回复片段。"""
        import json as _j

        if event.get("type") != "assistant":
            return
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                args = _j.dumps(block.get("input") or {}, ensure_ascii=False)
                click.echo(f"    · CC→ {block.get('name')}({args[:90]})")
            elif block.get("type") == "text" and (block.get("text") or "").strip():
                click.echo(f"    · CC: {(block.get('text') or '')[:90]!r}")

    def factory():
        if adapter == "openai-loop":
            from ahedd.adapters.openai_loop import OpenAILoopAdapter

            chat_kwargs: dict = {"max_tokens": max_tokens}
            if disable_thinking:
                chat_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            return OpenAILoopAdapter(make_client(agent_spec), chat_kwargs=chat_kwargs)
        if adapter == "deepagents":
            from ahedd.adapters.deepagents_adapter import DeepAgentsAdapter

            return DeepAgentsAdapter(
                agent_spec,
                disable_thinking=disable_thinking,
                tool_mode=tool_mode,
                mcp_url=_mcp_url,
                events_file=_mcp_events_file,
            )
        if adapter == "tau":
            from ahedd.adapters.tau_adapter import TauAdapter

            return TauAdapter(agent_spec)
        if adapter == "claude-code":
            from ahedd.adapters.claude_code_adapter import ClaudeCodeAdapter

            return ClaudeCodeAdapter(
                workdir=os.environ.get("AHEDD_CC_DIR", "."),
                ssh_target=os.environ.get("AHEDD_CC_SSH") or None,
                node_bin=os.environ.get("AHEDD_CC_NODE_BIN", "~/.nvm/versions/node/v22.22.0/bin"),
                timeout=cc_timeout,
                tool_mode=tool_mode,
                mcp_url=_mcp_url,
                events_file=_mcp_events_file,
                events_ssh_target=_mcp_events_ssh,
                on_event=_cc_live_print,
            )
        raise click.UsageError(f"未知 adapter: {adapter!r}（见 ahedd adapters list）")

    def _print_result(case, outcome, trial: int) -> None:
        """每完成一条立即输出（流式进度）。"""
        violations = check_trajectory_rules(case, outcome.trajectory)
        ok = not violations and outcome.stop_reason == "stop" and outcome.error is None
        meta = outcome.trajectory.meta
        tok = f"{meta.total_usage.input_tokens}/{meta.total_usage.output_tokens}"
        trial_tag = f"[{trial}/{trials}] " if trials > 1 else ""
        trace_path = f"runs/{meta.domain}/{case.id}/{meta.run_id}.jsonl"
        click.echo(
            f"[{'PASS' if ok else 'FAIL'}] {trial_tag}{case.id} "
            f"stop={outcome.stop_reason} steps={len(outcome.trajectory.steps)} "
            f"tokens(in/out)={tok} "
            f"env_changed={'yes' if outcome.env_diff else 'no'} "
            f"violations={violations or '-'}"
        )
        click.echo(f"        reply: {outcome.final_message[:120]!r} trace: {trace_path}")
        if outcome.env_diff:
            click.echo(f"        env_diff: {_json.dumps(outcome.env_diff, ensure_ascii=False, default=str)[:200]}")

    if tool_mode == "mcp" and adapter not in ("claude-code", "deepagents"):
        raise click.UsageError("--tool-mode mcp 当前支持 claude-code 与 deepagents 车道")

    _mcp_events_file: str | None = None
    _mcp_events_ssh: str | None = None
    _cleanup_procs: list = []
    _cleanup_cmds: list[str] = []
    _case_setup = None

    def _wait_local_port(port: int) -> None:
        import socket
        import time as _time

        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                return
            except OSError:
                _time.sleep(0.5)
        raise click.ClickException(f"MCP server 未能在端口 {port} 就绪")

    def _wait_remote_ready(target: str, port: int) -> None:
        import subprocess

        ready = subprocess.run(
            ["tsh", "ssh", target,
             f"for i in $(seq 1 40); do ss -tln | grep -q ':{port} ' && exit 0; sleep 0.5; done; exit 1"],
            capture_output=True, timeout=90, check=False,
        )
        if ready.returncode != 0:
            subprocess.run(["tsh", "ssh", target, "cat /tmp/ahedd_mcp.log"], check=False)
            raise click.ClickException(
                f"远端 MCP server 未就绪（{target}:{port}）：确认已装本包且 AHEDD_CC_PYTHON 正确"
            )

    try:
        if tool_mode == "mcp":
            import shlex as _shlex
            import subprocess
            import sys
            from pathlib import Path

            _ssh_target = os.environ.get("AHEDD_CC_SSH")
            if adapter == "claude-code" and _ssh_target:
                # 远端拓扑：MCP server 与 CC 同机（同主机才能走 localhost）。
                # （部分跳板方案不支持 ssh -R 反向转发，故远端自起 server 是通用做法）
                _remote_python = os.environ.get("AHEDD_CC_PYTHON", "python")
                _remote_events = "/tmp/ahedd_mcp_events.jsonl"
                _mcp_events_file = _remote_events
                _mcp_events_ssh = _ssh_target
                _cleanup_cmds.append(f"pkill -f 'ahedd[.]mcp.*--port {mcp_port}' || true")

                _domain_flag = f"--domain {domain} " if domain else ""

                def _case_setup(case):
                    # 杀进程与起进程必须分两次 ssh：合并成一条时，命令行里 nohup 部分的
                    # "ahedd.mcp ... --port" 字面量会被 pkill -f 匹配到自身 shell 而整条自杀
                    subprocess.run(
                        ["tsh", "ssh", _ssh_target,
                         f"pkill -f 'ahedd[.]mcp.*--port {mcp_port}' || true"],
                        capture_output=True, timeout=30, check=False,
                    )
                    subprocess.run(
                        ["tsh", "ssh", _ssh_target,
                         (f"rm -f {_remote_events} /tmp/ahedd_mcp.log; "
                          f"nohup {_shlex.quote(_remote_python)} -m ahedd.mcp --dataset {dataset} "
                          f"{_domain_flag}--http --port {mcp_port} --events-file {_remote_events} "
                          f"--env-seed {getattr(case, 'env_seed', 0) or 0} "
                          ">/tmp/ahedd_mcp.log 2>&1 &")],
                        capture_output=True, timeout=60, check=False,
                    )
                    _wait_remote_ready(_ssh_target, mcp_port)

                _case_setup(None)  # 首个 case 前也确保就绪
                click.echo(f"# mcp server (remote, per-case restart): {_ssh_target}:{mcp_port}")
            else:
                _mcp_events_file = str(Path("runs") / "mcp_events.jsonl")
                Path(_mcp_events_file).parent.mkdir(parents=True, exist_ok=True)
                _server_proc: subprocess.Popen | None = None

                def _case_setup(case):
                    nonlocal _server_proc
                    if _server_proc is not None:
                        _server_proc.terminate()
                        _server_proc.wait(timeout=10)
                    Path(_mcp_events_file).unlink(missing_ok=True)
                    _server_proc = subprocess.Popen(
                        [sys.executable, "-m", "ahedd.mcp", "--dataset", dataset,
                         "--http", "--port", str(mcp_port), "--events-file", _mcp_events_file,
                         "--env-seed", str(getattr(case, "env_seed", 0) or 0)]
                    )
                    _cleanup_procs.append(_server_proc)
                    _wait_local_port(mcp_port)

                _case_setup(None)
                click.echo(f"# mcp server (local, per-case restart): port={mcp_port}")

        def _sim_factory(case):
            from ahedd.llm import make_client as _mc
            from ahedd.user import UserSimulator

            sim_kwargs: dict = {"max_tokens": 512}
            if not think_sim:
                sim_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            return UserSimulator(_mc(roles.user_simulator), case, chat_kwargs=sim_kwargs)

        use_sim = roles.user_simulator is not None and not no_user_sim
        if use_sim:
            think_sim = not disable_thinking
            click.echo(f"# user simulator: {roles.user_simulator.model}（多轮对话模式）")

        pairs = asyncio.run(
            run_dataset(
                provider=get_dataset(dataset),
                adapter_factory=factory,
                domain=domain,
                case_ids=case_ids.split(",") if case_ids else None,
                trials=trials,
                dataset=dataset,
                agent_model=roles.agent.model,
                concurrency=1 if tool_mode == "mcp" else concurrency,  # per-case server 重启需串行
                on_result=_print_result,
                case_setup=_case_setup,
                user_simulator_factory=_sim_factory if use_sim else None,
            )
        )
    finally:
        for proc in _cleanup_procs:
            proc.terminate()
        for cmd in _cleanup_cmds:
            ssh = os.environ.get("AHEDD_CC_SSH")
            if ssh:
                subprocess.run(["tsh", "ssh", ssh, cmd], capture_output=True, timeout=60, check=False)

    n_pass = sum(
        1
        for case, outcome in pairs
        if not check_trajectory_rules(case, outcome.trajectory)
        and outcome.stop_reason == "stop"
        and outcome.error is None
    )
    click.echo(f"# {n_pass}/{len(pairs)} runs OK (deterministic channel; run `ahedd score` for rubric judging)")


@main.command("score")
@click.option("--runs", "runs_dir", default="runs", show_default=True, help="轨迹目录")
@click.option("--dataset", default="mock", show_default=True, help="用于取 TaskCase/rubrics 的数据集")
@click.option("--models", "models_yaml", default=None, help="models.yaml 路径（缺省用 .env/环境变量）")
@click.option("--no-judge", is_flag=True, help="只算确定性指标，跳过 LLM 判分")
@click.option("--think", is_flag=True, help="判分模型开启思考（默认关闭以保证 JSON 输出）")
@click.option("--strict-state", is_flag=True, help="强制开启终态断言（expected_states 字段级比对；官方基准默认关闭以对齐论文口径）")
@click.option("--no-state-check", is_flag=True, help="强制关闭终态断言")
def score_cmd(runs_dir: str, dataset: str, models_yaml: str | None, no_judge: bool, think: bool, strict_state: bool, no_state_check: bool) -> None:
    """离线判分：读轨迹 -> rubric 滑窗 judge + 轨迹动力学指标 -> 结果落盘（先存后判）。"""
    import asyncio
    import json as _json
    from pathlib import Path

    from ahedd.config import load_models_config
    from ahedd.datasets import get_dataset
    from ahedd.llm import make_client
    from ahedd.scoring import (
        RubricSlidingWindowScorer,
        check_expected_states,
        compute_trajectory_metrics,
        summarize_suite,
    )
    from ahedd.trace.schema import load_jsonl_trajectory

    provider = get_dataset(dataset)
    scorer = None
    if not no_judge:
        roles = load_models_config(models_yaml)
        if roles.judge is None:
            raise click.UsageError("判分器未配置：请设置 AHEDD_JUDGE_* 环境变量，或使用 --no-judge")
        chat_kwargs: dict = {"max_tokens": 2048}
        if not think:
            chat_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        scorer = RubricSlidingWindowScorer(make_client(roles.judge), chat_kwargs=chat_kwargs)

    traces = sorted(Path(runs_dir).rglob("*.jsonl"))
    if not traces:
        raise click.UsageError(f"{runs_dir} 下没有轨迹文件")
    click.echo(f"# scoring {len(traces)} traces (judge={'off' if no_judge else 'on'})")

    suite_rows: list[tuple[str, str, bool | None, object]] = []
    for trace_file in traces:
        try:
            trajectory = load_jsonl_trajectory(str(trace_file))
            if not trajectory.meta.task_id:
                raise ValueError("not a trajectory file")
        except Exception:  # noqa: BLE001 - 跳过非轨迹 JSONL（如 MCP server 事件日志）
            click.echo(f"[SKIP] {trace_file.name}: 非轨迹文件")
            continue
        meta = trajectory.meta
        case = next(
            (c for c in provider.load(meta.domain) if c.id == meta.task_id), None
        )
        if case is None:
            click.echo(f"[SKIP] {meta.task_id} 不在数据集 {dataset} 的 {meta.domain} 域中")
            continue

        passed: bool | None = None
        score_value = None
        rubric_detail: list = []
        state_violations: list = []
        if scorer is not None and case.rubrics:
            report = asyncio.run(scorer.score(case, trajectory))
            # 终态断言策略：显式 flag > 数据集策略
            # vita（官方基准）默认关闭以对齐论文口径（官方 evaluator 不消费 final_state）；
            # 自定义数据集默认开启（expected_states 由数据集作者自行定义，属增强保障）
            _state_on = strict_state if strict_state else (not no_state_check and dataset != "vita")
            state_file = trace_file.with_suffix(".envstate.json")
            ec = (getattr(case, "extra", None) or {}).get("evaluation_criteria") or {}
            expected = ec.get("expected_states") if isinstance(ec, dict) else None
            if _state_on and state_file.exists() and expected:
                import json as _j2

                try:
                    final_state = _j2.loads(state_file.read_text(encoding="utf-8"))
                    state_violations = check_expected_states(expected, final_state)
                except Exception:  # noqa: BLE001 - 断言失败不阻断判分
                    state_violations = []
            report.state_violations = state_violations
            report.passed = report.passed and not state_violations
            passed = report.passed
            score_value = report.score
            rubric_detail = [r.model_dump() for r in report.rubric_results]
        metrics = compute_trajectory_metrics(trajectory)

        status = "----" if passed is None else ("PASS" if passed else "FAIL")
        rubric_str = f"{sum(r['satisfied'] for r in rubric_detail)}/{len(rubric_detail)}" if rubric_detail else "-"
        click.echo(
            f"[{status}] {meta.task_id} adapter={meta.adapter} "
            f"rubric={rubric_str} score={score_value if score_value is not None else '-'} "
            f"turns={metrics.turns} useful={metrics.useful_action_ratio} "
            f"errs={metrics.error_events} tokens={metrics.tokens_in}/{metrics.tokens_out}"
            + (f" state_violations={len(state_violations)}" if state_violations else "")
        )
        suite_rows.append((meta.adapter, meta.task_id, passed, metrics))

        out = trace_file.with_suffix(".score.json")
        out.write_text(
            _json.dumps(
                {
                    "trace": str(trace_file),
                    "passed": passed,
                    "score": score_value,
                    "rubric_results": rubric_detail,
                    "state_violations": state_violations,
                    "metrics": metrics.model_dump(),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    click.echo("\n# suite summary（Step-to-Success / 错误恢复 / 帕累托原料）")
    click.echo(_json.dumps(summarize_suite(suite_rows), ensure_ascii=False, indent=1))


@main.command("report")
@click.option("--runs", "runs_dir", default="runs", show_default=True, help="轨迹目录")
@click.option("--out", default="report.html", show_default=True)
def report_cmd(runs_dir: str, out: str) -> None:
    """生成单文件 HTML 诊断报告（轨迹回放 + rubric 判分 + 环境 diff）。"""
    import json as _json
    from pathlib import Path

    from ahedd.report.html import render_report
    from ahedd.trace.schema import load_jsonl_trajectory

    items = []
    for trace_file in sorted(Path(runs_dir).rglob("*.jsonl")):
        try:
            t = load_jsonl_trajectory(str(trace_file))
            if not t.meta.task_id:
                continue
        except Exception:  # noqa: BLE001, S112 - 跳过事件日志等非轨迹文件
            continue
        score = None
        sidecar = trace_file.with_suffix(".score.json")
        if sidecar.exists():
            try:
                score = _json.loads(sidecar.read_text(encoding="utf-8"))
            except _json.JSONDecodeError:
                score = None
        env_diff = {}
        diff_sidecar = trace_file.with_suffix(".envdiff.json")
        if diff_sidecar.exists():
            try:
                env_diff = _json.loads(diff_sidecar.read_text(encoding="utf-8"))
            except _json.JSONDecodeError:
                env_diff = {}
        items.append(
            {
                "meta": t.meta.model_dump(),
                "steps": [s.model_dump(exclude_none=True) for s in t.steps],
                "score": score,
                "env_diff": env_diff,
            }
        )
    if not items:
        raise click.UsageError(f"{runs_dir} 下没有轨迹文件")
    out_path = render_report(items, out)
    click.echo(f"report -> {out_path} ({len(items)} traces)")


@main.command("freeze")
@click.argument("trace", default="")
@click.option("--dataset", default=None, help="缺省取轨迹 meta.dataset")
@click.option("--attribution", default=None, help="人工归因标签（缺省自动预判）")
def freeze_cmd(trace: str, dataset: str | None, attribution: str | None) -> None:
    """把失败轨迹冻结为回归用例（regressions/cases/RC-xxxx.yaml + 基线轨迹副本）。"""
    import datetime as _dt
    import shutil
    from pathlib import Path

    import yaml as _yaml

    from ahedd.datasets import get_dataset
    from ahedd.regression.schema import Attribution, RegressionCase
    from ahedd.scoring.trajectory_metrics import compute_trajectory_metrics
    from ahedd.trace.schema import load_jsonl_trajectory

    p = _locate_trace(trace)
    t = load_jsonl_trajectory(str(p))
    ds = dataset or t.meta.dataset or "mock"
    provider = get_dataset(ds)
    case = next((c for c in provider.load(t.meta.domain) if c.id == t.meta.task_id), None)
    if case is None:
        raise click.UsageError(f"任务 {t.meta.task_id} 不在数据集 {ds} 的 {t.meta.domain} 域中")

    metrics = compute_trajectory_metrics(t)
    label = attribution or _triage(t, metrics)

    cases_dir = Path("regressions/cases")
    cases_dir.mkdir(parents=True, exist_ok=True)
    existing = list(cases_dir.glob("RC-*.yaml"))
    next_id = f"RC-{len(existing) + 1:04d}"

    trace_copy = Path("regressions/traces") / t.meta.task_id / f"{t.meta.run_id}.jsonl"
    trace_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, trace_copy)

    rc = RegressionCase(
        id=next_id,
        source_run=str(p),
        domain=t.meta.domain,
        dataset=ds,
        env_seed=case.env_seed,
        instruction=case.instruction,
        user_scenario=case.user_scenario.model_dump() if case.user_scenario else None,
        rubrics=case.rubrics,
        rules=case.rules.model_dump(),
        attribution=Attribution(primary=label),
        baseline_trace=str(trace_copy).replace("\\", "/"),
        created_by="human" if attribution else "auto",
        created_at=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    out = cases_dir / f"{next_id}.yaml"
    out.write_text(_yaml.safe_dump(rc.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    click.echo(f"frozen -> {out} (attribution={label}, baseline={trace_copy})")


def _locate_trace(trace: str) -> Path:
    """轨迹定位：路径直接用；否则按 run_id / case_id 在 runs/ 下搜（取最新）。"""
    from pathlib import Path

    if trace:
        p = Path(trace)
        if p.exists():
            return p
        hits = sorted(
            (f for f in Path("runs").rglob(f"*{trace}*.jsonl")
             if not f.name.endswith((".score.json", ".envdiff.json"))),
            key=lambda f: f.stat().st_mtime,
        )
        if not hits:
            raise click.UsageError(f"找不到轨迹：{trace}")
        return hits[-1]
    hits = sorted(
        (f for f in Path("runs").rglob("*.jsonl")
         if not f.name.endswith((".score.json", ".envdiff.json"))),
        key=lambda f: f.stat().st_mtime,
    )
    if not hits:
        raise click.UsageError("runs/ 下没有轨迹")
    return hits[-1]


def _triage(t: Any, metrics: Any) -> str:
    """自动归因预判（人工可用 --attribution 覆盖）。"""
    errors = [s for s in t.steps if s.type == "error"]
    if any("loop" in (s.stop_reason or "") or "loop detected" in s.content for s in errors):
        return "tool.loop"
    if any(s.stop_reason == "max_steps" for s in errors):
        return "reasoning.unfinished"
    if metrics.malformed_args_calls > 0 or metrics.unknown_tool_calls > 0:
        return "tool.param"
    if metrics.failed_calls > 0 or metrics.error_events > 0:
        return "tool"
    return "reasoning.constraint"


@main.group("mcp")
def mcp_cmd() -> None:
    """MCP server 工具集：把数据集环境暴露给 MCP 客户端被测对象。"""


@mcp_cmd.command("serve")
@click.option("--dataset", default="mock", show_default=True)
@click.option("--domain", default=None)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8023, show_default=True)
@click.option("--stdio", is_flag=True, help="stdio 传输（默认 streamable-http）")
@click.option("--events-file", default=None, help="工具/环境事件日志（JSONL）")
def mcp_serve_cmd(dataset: str, domain: str | None, host: str, port: int, stdio: bool, events_file: str | None) -> None:
    """运行环境的 MCP server（阻塞；Ctrl-C 退出）。"""
    from ahedd.mcp.server import run_server

    run_server(
        dataset,
        domain=domain,
        transport="stdio" if stdio else "streamable-http",
        host=host,
        port=port,
        events_path=events_file,
    )


@main.command("ci")
def ci_cmd() -> None:
    """回归门禁：全量回归用例红绿检查。"""
    click.echo("脚手架阶段：ahedd ci 于 D4 里程碑实现。")


if __name__ == "__main__":
    main()
