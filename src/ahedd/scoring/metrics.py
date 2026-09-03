"""指标：Avg@k / Pass@k / Pass^k（含无偏组合估计）与分环节指标聚合。

Pass@k 采用 HumanEval 式无偏估计；Pass^k 为 k 次全成功概率。
统计依据见调研报告 §2.6：k=4 为精度/成本平衡点。
"""

from __future__ import annotations

from math import comb
from statistics import fmean


def avg_k(scores: list[float]) -> float:
    """多次运行的平均得分（主榜单指标 Avg@k，k=len(scores)）。"""
    if not scores:
        raise ValueError("avg_k requires at least one score")
    return fmean(scores)


def pass_at_k(n_success: int, n_trials: int, k: int) -> float:
    """P(k 次独立试验中至少 1 次成功) 的无偏估计（超几何）。

    要求 n_trials >= k >= 1 且 0 <= n_success <= n_trials。
    """
    if not (1 <= k <= n_trials):
        raise ValueError(f"need 1 <= k <= n_trials, got k={k}, n_trials={n_trials}")
    n_success = max(0, min(n_success, n_trials))
    return 1.0 - comb(n_trials - n_success, k) / comb(n_trials, k)


def pass_all_k(n_success: int, n_trials: int, k: int) -> float:
    """P(k 次独立试验全部成功)（Pass^k，生产可靠性视角）。"""
    if not (1 <= k <= n_trials):
        raise ValueError(f"need 1 <= k <= n_trials, got k={k}, n_trials={n_trials}")
    n_success = max(0, min(n_success, n_trials))
    if n_success < k:
        return 0.0
    return comb(n_success, k) / comb(n_trials, k)


def aggregate(scores: list[float]) -> dict[str, float]:
    """任务级多次运行 -> 单任务指标包。"""
    n = len(scores)
    successes = sum(1 for s in scores if s >= 1.0)
    return {
        "avg_k": avg_k(scores),
        f"pass@{n}": pass_at_k(successes, n, n),
        f"pass^{n}": pass_all_k(successes, n, n),
    }
