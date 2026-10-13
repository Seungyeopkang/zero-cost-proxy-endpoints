import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas import metrics, search
from exnas.space import N_EDGES


def test_kl_uniform_is_near_zero():
    rng = np.random.default_rng(0)
    pool = search.random_pool(20000, rng)
    assert metrics.population_kl(pool) < 0.02


def test_kl_maximal_for_degenerate_population():
    pool = [(0,) * N_EDGES] * 100
    assert metrics.population_kl(pool) == pytest_approx(metrics.MAX_KL)


def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol

        def __repr__(self):
            return f"~{x}"

    return _A()


def test_kl_monotone_in_selection_pressure():
    """선택압이 셀수록 KL이 커진다 — 최적화 강도 축의 타당성 확인."""
    fit = lambda a: float(sum(1 for o in a if o == 3))  # noqa: E731
    kls = []
    for sample_size in (2, 10, 40):
        hist = search.regularized_evolution(
            2000, fit, np.random.default_rng(0), population_size=50, sample_size=sample_size
        )
        kls.append(metrics.population_kl(hist))
    assert kls[0] < kls[1] < kls[2], kls


def test_tau_extremes():
    a = np.arange(100.0)
    assert metrics.tau(a, a) == 1.0
    assert metrics.tau(a, -a) == -1.0
    assert np.isnan(metrics.tau(np.array([1.0]), np.array([1.0])))


def test_selection_gap_zero_when_best_selected():
    pool = np.array([1.0, 5.0, 3.0])
    assert metrics.selection_gap(pool, np.array([5.0])) == 0.0
    assert metrics.selection_gap(pool, np.array([3.0])) == 2.0
