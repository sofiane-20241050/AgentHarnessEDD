"""Bad Case -> 回归用例资产（回流层）。

沉淀形态（见调研报告 §6.2）：一条冻结的失败 = 环境种子 + 用户剧本
+ rubric 断言 + 归因标签 + 修复前红基线轨迹，随 git 版本化。
经验 = 通过回归门禁的那次 diff + 归因统计的下降曲线。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Attribution(BaseModel):
    """归因标签：一级对齐论文三维框架，二级为可操作标签。"""

    primary: str  # reasoning | tool | interaction | noise
    secondary: str = ""  # 如 tool.param.value / reasoning.constraint_dropped / interaction.no_clarify
    detail: str = ""


class RegressionCase(BaseModel):
    """regressions/cases/RC-xxxx.yaml 的数据模型。"""

    id: str
    source_run: str            # 首次失败轨迹的溯源（runs/<date>/<domain>/<task>/<run_id>）
    domain: str = ""
    dataset: str = ""
    env_seed: int | None = None
    instruction: str = ""
    user_scenario: dict | None = None
    rubrics: list[str] = Field(default_factory=list)
    rules: dict = Field(default_factory=dict)
    attribution: Attribution | None = None
    baseline_trace: str = ""   # 修复前"红测试"基线轨迹文件
    created_by: str = "auto"   # auto（自动分诊）| human（人工确认）
    created_at: str = ""
