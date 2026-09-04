"""D4 测试：HTML 报告渲染 + freeze 回归冻结（含自动归因预判）。"""

import json

from click.testing import CliRunner

from ahedd.cli import main
from ahedd.report.html import render_report


def _write_fake_run(root, task_id="mock_001_change_address", adapter="openai-loop", run_id="r1"):
    """在 root/runs/... 下造一条失败轨迹 + 旁车文件，返回轨迹路径。"""
    trace = root / "runs" / "mock" / task_id / f"{run_id}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "task_id": task_id, "domain": "mock", "dataset": "mock",
            "adapter": adapter, "agent_model": "m", "started_at": "2026-09-04T00:00:00",
            "total_usage": {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.0}}
    steps = [
        {"index": 0, "type": "user", "content": "改地址", "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}, "ts": "2026-09-04T00:00:01"},
        {"index": 1, "type": "error", "content": "loop detected: update_address x11 identical calls (circuit breaker)",
         "tool_name": "update_address", "error_kind": "agent", "stop_reason": "loop_detected",
         "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}, "ts": "2026-09-04T00:00:02"},
    ]
    trace.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [meta, *steps]) + "\n", encoding="utf-8")
    trace.with_suffix(".score.json").write_text(json.dumps({
        "trace": str(trace), "passed": False, "score": 0.0,
        "rubric_results": [
            {"key": "r0", "description": "地址最终为北京市朝阳区", "satisfied": False, "evidence_turn": None},
        ],
        "metrics": {},
    }, ensure_ascii=False), encoding="utf-8")
    trace.with_suffix(".envdiff.json").write_text("{}", encoding="utf-8")
    return trace


def test_render_report(tmp_path):
    trace = _write_fake_run(tmp_path, task_id="t_report")
    items = [{
        "meta": json.loads(trace.read_text(encoding="utf-8").splitlines()[0]),
        "steps": [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines()[1:]],
        "score": json.loads(trace.with_suffix(".score.json").read_text(encoding="utf-8")),
        "env_diff": {},
    }]
    out = render_report(items, str(tmp_path / "report.html"))
    from pathlib import Path as _P
    html = _P(out).read_text(encoding="utf-8")
    assert "t_report" in html
    assert "地址最终为北京市朝阳区" in html  # rubric 明细内嵌
    assert "loop detected" in html


def test_freeze_creates_regression_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = _write_fake_run(tmp_path)
    result = CliRunner().invoke(main, ["freeze", str(trace)])
    assert result.exit_code == 0, result.output
    rc_file = tmp_path / "regressions" / "cases" / "RC-0001.yaml"
    assert rc_file.exists()
    content = rc_file.read_text(encoding="utf-8")
    assert "tool.loop" in content          # 自动归因预判命中死循环
    assert "created_by: auto" in content
    assert (tmp_path / "regressions" / "traces" / "mock_001_change_address" / "r1.jsonl").exists()  # 基线轨迹副本

    # 第二条冻结应递增编号
    trace2 = _write_fake_run(tmp_path, task_id="mock_002_reject_cancel_shipped", run_id="r2")
    result2 = CliRunner().invoke(main, ["freeze", str(trace2), "--attribution", "tool.param"])
    assert result2.exit_code == 0, result2.output
    assert (tmp_path / "regressions" / "cases" / "RC-0002.yaml").exists()
    assert "tool.param" in (tmp_path / "regressions" / "cases" / "RC-0002.yaml").read_text(encoding="utf-8")
    assert "created_by: human" in (tmp_path / "regressions" / "cases" / "RC-0002.yaml").read_text(encoding="utf-8")
