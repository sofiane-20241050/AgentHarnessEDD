"""单文件 HTML 诊断报告。

目标形态（Playwright Trace Viewer 式体验，零依赖单文件）：
  - 轨迹时间轴回放（每轮 Thought / Action / Tool I/O）
  - rubric 状态演化 + 违例标注
  - 环境状态 diff（执行前后，Git-Diff 风格高亮）
  - 失败归因标签（死循环 / schema 错误 / 越权调用 / 未澄清 ...）
  - 指标汇总（Avg@k / Pass@k / Pass^k、轮次、token、成本）
D4 里程碑实现（jinja2 内联模板 + 内嵌 JSON 数据）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahedd.trace.schema import Trajectory


def render_report(trajectories: list[Trajectory], out_path: str) -> str:
    """渲染并写出单文件 HTML 报告，返回文件路径。"""
    raise NotImplementedError("HTML 诊断报告于 D4 里程碑实现")
