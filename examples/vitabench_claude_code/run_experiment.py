#!/usr/bin/env python3
"""端到端实战：Claude Code CLI × VitaBench（MCP 原生工具 + 用户模拟器 + 判分 + 报告）。

用法（解释器随意——脚本自动优先使用项目 .venv，避免误用系统 Python 踩依赖/编码坑）::

    python run_experiment.py --cases 10711002
    python run_experiment.py --cases 10711002,10711003 --skip-report

前置条件见同目录 README.md。脚本按序调用框架 CLI（与手动执行完全等价、编排单源维护）：
run（含 MCP server 自动拉起/按 case 重启）→ score → report。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def pick_python() -> str:
    """优先项目 .venv 解释器（依赖齐全、编码可控），缺省回退当前解释器。"""
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_step(title: str, cli_args: list[str], env: dict) -> int:
    print(f"{title}: ahedd {' '.join(cli_args)}", flush=True)
    proc = subprocess.run(
        [env["__PYTHON__"], "-m", "ahedd.cli", *cli_args],
        cwd=str(ROOT), env={k: v for k, v in env.items() if k != "__PYTHON__"}, check=False,
    )
    if proc.returncode != 0:
        print(f"    (exit {proc.returncode})")
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code CLI × VitaBench 端到端实验")
    parser.add_argument("--cases", required=True, help="逗号分隔的 case id，如 10711002")
    parser.add_argument("--domain", default="delivery", help="vita 域（默认 delivery）")
    parser.add_argument("--adapter", default="claude-code", help="被测车道（默认 claude-code）")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--no-user-sim", action="store_true", help="禁用用户模拟器")
    args = parser.parse_args()

    python = pick_python()
    print(f"[解释器] {python}")

    if SRC.exists():  # 允许未安装（-e）时从源码运行
        sys.path.insert(0, str(SRC))
    from ahedd.config import load_dotenv

    load_dotenv(ROOT / ".env")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    env["__PYTHON__"] = python

    common = ["--dataset", "vita", "--domain", args.domain]
    runs_dir = f"runs/{args.domain}"

    run_step("[1/3] 跑评测（自动拉起 MCP server + 用户模拟器多轮）",
             ["run", *common, "--adapter", args.adapter, "--tool-mode", "mcp",
              "--cases", args.cases, *(["--no-user-sim"] if args.no_user_sim else [])], env)
    run_step("[2/3] 离线判分（rubric 滑窗 judge + 过程指标）",
             ["score", "--runs", runs_dir, "--dataset", "vita"], env)
    if not args.skip_report:
        run_step("[3/3] HTML 诊断报告",
                 ["report", "--runs", runs_dir, "--out", "vita_report.html"], env)

    print("\n产物：runs/（轨迹+判分+diff）、vita_report.html；失败轨迹冻结：ahedd freeze <run_id>")


if __name__ == "__main__":
    main()
