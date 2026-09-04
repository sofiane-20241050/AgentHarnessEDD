"""AgentHarnessEDD —— 面向 Agent Harness 的评估驱动开发（EDD）框架。

分层结构（见 docs/research-report.md §4）：
  datasets   数据集接入（任务 + rubric + 用户剧本）
  env        仿真环境（工具 + 虚拟数据库，确定性内核）
  adapters   被测 Agent 三车道接入（进程内 / MCP / RPC）
  trace      统一轨迹 Schema 与录制器
  scoring    判分双通道（rubric LLM judge + 确定性断言）与指标
  regression Bad Case -> 回归用例资产
  report     单文件 HTML 诊断报告
"""

from __future__ import annotations

import os

# Windows 中文区默认编码为 GBK：第三方数据集代码常直接 open() 不带 encoding，
# 读 UTF-8 数据会 UnicodeDecodeError。置 PYTHONUTF8 使本框架拉起的子进程
# （如本地 MCP server）进入 UTF-8 模式；当前进程内的兼容见 vita 插件的 _force_utf8_open。
os.environ.setdefault("PYTHONUTF8", "1")

__version__ = "0.1.0"
