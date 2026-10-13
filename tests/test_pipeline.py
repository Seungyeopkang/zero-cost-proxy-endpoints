import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas.aggregate import RandomProxy, SingleProxy
from exnas.benchmark import SyntheticBenchmark
from exnas.pipeline import make_calibrator, run_pipeline


@pytest.fixture(scope="module")
def bench():
    return SyntheticBenchmark(seed=0, n_proxies=4)


@pytest.fixture(scope="module")
def cal(bench):
    return make_calibrator(bench, np.random.default_rng(0), n_calib=1000)


def test_budget_cannot_exceed_candidates(bench, cal):
    with pytest.raises(ValueError):
        run_pipeline(bench, SingleProxy(0), cal, "random", 10, 20, seed=0)


def test_run_is_deterministic_given_seed(bench, cal):
    a = run_pipeline(bench, SingleProxy(0), cal, "ea", 300, 10, seed=3)
    b = run_pipeline(bench, SingleProxy(0), cal, "ea", 300, 10, seed=3)
    assert a.to_dict() == b.to_dict()


def test_ea_has_higher_kl_than_random(bench, cal):
    r = run_pipeline(bench, SingleProxy(0), cal, "random", 1000, 20, seed=0)
    e = run_pipeline(bench, SingleProxy(0), cal, "ea", 1000, 20, seed=0)
    assert e.population_kl > r.population_kl


def test_n_equals_budget_means_no_filtering(bench, cal):
    """N=B이면 proxy가 결과에 영향을 줄 수 없다 — 어떤 결합기든 같은 답."""
    a = run_pipeline(bench, SingleProxy(0), cal, "random", 40, 40, seed=1)
    b = run_pipeline(bench, RandomProxy(), cal, "random", 40, 40, seed=1)
    assert a.best_acc == b.best_acc


def test_selected_never_beats_pool_oracle(bench, cal):
    r = run_pipeline(bench, SingleProxy(0), cal, "random", 500, 20, seed=0)
    assert r.best_acc <= r.pool_best_acc + 1e-9
    assert r.selection_gap >= -1e-9


def test_regret_matches_best_acc(bench, cal):
    r = run_pipeline(bench, SingleProxy(0), cal, "random", 500, 20, seed=0)
    assert r.regret == pytest.approx(bench.best_acc - r.best_acc)
