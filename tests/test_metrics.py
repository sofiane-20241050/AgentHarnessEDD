"""指标纯函数测试：Avg@k / Pass@k / Pass^k。"""

import pytest

from ahedd.scoring.metrics import aggregate, avg_k, pass_all_k, pass_at_k


def test_avg_k() -> None:
    assert avg_k([1.0, 0.0, 1.0, 1.0]) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        avg_k([])


def test_pass_at_k_bounds() -> None:
    assert pass_at_k(0, 4, 4) == 0.0
    assert pass_at_k(1, 4, 4) == 1.0  # 至少一次成功 => 必然
    with pytest.raises(ValueError):
        pass_at_k(1, 4, 5)  # k 不能超过试验数


def test_pass_at_k_unbiased() -> None:
    # 4 次试验 2 次成功，k=2：1 - C(2,2)/C(4,2) = 1 - 1/6
    assert pass_at_k(2, 4, 2) == pytest.approx(1 - 1 / 6)


def test_pass_all_k() -> None:
    assert pass_all_k(4, 4, 4) == 1.0
    assert pass_all_k(3, 4, 4) == 0.0  # 成功数不足 k 必为 0
    # C(3,2)/C(4,2) = 3/6
    assert pass_all_k(3, 4, 2) == pytest.approx(0.5)


def test_aggregate() -> None:
    m = aggregate([1.0, 1.0, 0.0, 0.0])
    assert m["avg_k"] == 0.5
    assert m["pass@4"] == 1.0
    assert m["pass^4"] == 0.0
