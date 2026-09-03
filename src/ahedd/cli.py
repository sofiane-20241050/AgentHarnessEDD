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
) -> None:
    """跑评测：采集轨迹 + 确定性断言（rubric 判分用 ahedd score 离线执行，先存后判）。"""
    import asyncio
    import json as _json

    from ahedd.config import load_models_config
    from ahedd.datasets import get_dataset
    from ahedd.llm import make_client
    from ahedd.runner import run_dataset
    from ahedd.scoring.deterministic import check_trajectory_rules

    roles = load_models_config(models_yaml)
    click.echo(
        f"# dataset={dataset} adapter={adapter} model={roles.agent.model} "
        f"trials={trials} concurrency={concurrency}"
    )

    # --max-tokens 对所有适配器生效：统一写进 agent spec
    agent_spec = roles.agent.model_copy(update={"max_tokens": max_tokens})

    def factory():
        if adapter == "openai-loop":
            from ahedd.adapters.openai_loop import OpenAILoopAdapter

            chat_kwargs: dict = {"max_tokens": max_tokens}
            if disable_thinking:
                chat_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            return OpenAILoopAdapter(make_client(agent_spec), chat_kwargs=chat_kwargs)
        if adapter == "deepagents":
            from ahedd.adapters.deepagents_adapter import DeepAgentsAdapter

            return DeepAgentsAdapter(agent_spec, disable_thinking=disable_thinking)
        if adapter == "tau":
            from ahedd.adapters.tau_adapter import TauAdapter

            return TauAdapter(agent_spec)
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

    pairs = asyncio.run(
        run_dataset(
            provider=get_dataset(dataset),
            adapter_factory=factory,
            domain=domain,
            case_ids=case_ids.split(",") if case_ids else None,
            trials=trials,
            dataset=dataset,
            agent_model=roles.agent.model,
            concurrency=concurrency,
            on_result=_print_result,
        )
    )

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
def score_cmd(runs_dir: str, dataset: str, models_yaml: str | None, no_judge: bool, think: bool) -> None:
    """离线判分：读轨迹 -> rubric 滑窗 judge + 轨迹动力学指标 -> 结果落盘（先存后判）。"""
    import asyncio
    import json as _json
    from pathlib import Path

    from ahedd.config import load_models_config
    from ahedd.datasets import get_dataset
    from ahedd.llm import make_client
    from ahedd.scoring import RubricSlidingWindowScorer, compute_trajectory_metrics, summarize_suite
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
        trajectory = load_jsonl_trajectory(str(trace_file))
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
        if scorer is not None and case.rubrics:
            report = asyncio.run(scorer.score(case, trajectory))
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
@click.option("--run", "run_id", required=True, help="run 目录或 run_id")
def report_cmd(run_id: str) -> None:
    """生成单文件 HTML 诊断报告。"""
    click.echo("脚手架阶段：ahedd report 于 D4 里程碑实现。")


@main.command("freeze")
@click.argument("run_id")
def freeze_cmd(run_id: str) -> None:
    """把失败轨迹冻结为回归用例（regressions/cases/RC-xxxx.yaml）。"""
    click.echo("脚手架阶段：ahedd freeze 于 D4 里程碑实现。")


@main.command("ci")
def ci_cmd() -> None:
    """回归门禁：全量回归用例红绿检查。"""
    click.echo("脚手架阶段：ahedd ci 于 D4 里程碑实现。")


if __name__ == "__main__":
    main()
