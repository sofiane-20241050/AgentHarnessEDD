"""判分层入口：双通道 + 指标。"""

from ahedd.scoring.base import RubricResult, Scorer, ScoreReport
from ahedd.scoring.deterministic import (
    check_expected_states,
    check_trajectory_rules,
    count_repeated_failures,
)
from ahedd.scoring.metrics import aggregate, avg_k, pass_all_k, pass_at_k
from ahedd.scoring.rubric import RubricSlidingWindowScorer
from ahedd.scoring.trajectory_metrics import (
    TrajectoryMetrics,
    compute_trajectory_metrics,
    summarize_suite,
)

__all__ = [
    "RubricResult",
    "RubricSlidingWindowScorer",
    "ScoreReport",
    "Scorer",
    "TrajectoryMetrics",
    "aggregate",
    "avg_k",
    "check_expected_states",
    "check_trajectory_rules",
    "compute_trajectory_metrics",
    "count_repeated_failures",
    "pass_all_k",
    "pass_at_k",
    "summarize_suite",
]
