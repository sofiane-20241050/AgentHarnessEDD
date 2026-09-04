"""数据集接入层基类。

接入一套评测数据集 = 提供两样东西：
  1. 任务集：初始指令 + 用户剧本 + rubric 原子断言（+ 可选环境种子/轨迹规则）
  2. 环境工厂：为每个域构建仿真环境（工具 + 虚拟数据库）

公开基准（如 VitaBench）与企业私有集走同一套契约，仅实现方不同。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ahedd.env.base import Environment


class UserScenario(BaseModel):
    """用户模拟器剧本：Agent 需要在多轮交互中"逼问"出来的信息全在这里。"""

    persona: str | None = None          # 人格：急躁 / 注重细节 / 依赖性强 ...
    instruction: str = ""               # 完整任务指令（模拟器掌握的"底牌"）
    known_info: list[str] = Field(default_factory=list)    # 被问到会说的信息
    unknown_info: list[str] = Field(default_factory=list)  # 被问到答"不知道"的信息
    traits: dict[str, str] = Field(default_factory=dict)   # 饮食禁忌/偏好等扩展属性


class TrajectoryRules(BaseModel):
    """确定性轨迹规则（判分的确定性通道）。"""

    forbidden_tools: list[str] = Field(default_factory=list)
    max_turns: int = 300
    max_identical_calls: int = 3  # 同一工具同一参数连续调用超过该次数视为死循环


class TaskCase(BaseModel):
    """单个评测任务：数据集的最小单元。"""

    id: str
    domain: str
    instruction: str                    # 首轮用户输入
    user_scenario: UserScenario | None = None
    rubrics: list[str] = Field(default_factory=list)  # 原子断言，如"预订500米内的素食餐厅"
    rules: TrajectoryRules = Field(default_factory=TrajectoryRules)
    env_seed: int | None = None
    source: str = ""                    # 数据来源标记：vitabench / private / ...
    extra: dict[str, Any] = Field(default_factory=dict)  # 域特定数据（如 vita 的 expected_states 确定性断言原料）


@runtime_checkable
class DatasetProvider(Protocol):
    """数据集接入契约。"""

    name: str

    def domains(self) -> list[str]:
        """可用域列表，如 ["delivery", "in-store", "ota", "cross"]。"""
        ...

    def load(self, domain: str) -> list[TaskCase]:
        """加载指定域的全部任务。"""
        ...

    def build_environment(self, domain: str) -> Environment:
        """构建指定域的仿真环境。"""
        ...
