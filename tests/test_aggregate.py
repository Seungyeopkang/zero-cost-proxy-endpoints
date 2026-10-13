import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas import aggregate as A


def _ctx(n=200, k=4, seed=0):
    rng = np.random.default_rng(seed)
    cal = A.Calibrator(rng.standard_normal((5000, k)))
    return cal, rng.standard_normal((n, k))


def test_calibrator_u_in_range():
    cal, s = _ctx()
    _, u = cal.transform(s)
    assert u.min() >= 1.0 / cal.n_calib - 1e-12
    assert u.max() <= 1.0


def test_calibrator_is_order_independent():
    """같은 아키텍처는 평가 집합이 달라져도 같은 점수를 받아야 한다 (EA 오염 방지)."""
    cal, s = _ctx()
    z1, u1 = cal.transform(s)
    z2, u2 = cal.transform(s[:10])
    assert np.allclose(z1[:10], z2)
    assert np.allclose(u1[:10], u2)


def test_aggregators_return_expected_shape():
    cal, s = _ctx()
    z, u = cal.transform(s)
    for agg in A.default_aggregators():
        v = agg(z, u)
        assert v.shape == (s.shape[0],)
        assert np.isfinite(v).all()


def test_anti_oracle_is_exact_negation():
    cal, s = _ctx()
    z, u = cal.transform(s)
    base = A.SingleProxy(0)
    assert np.allclose(A.AntiOracle(base)(z, u), -base(z, u))


def test_consistency_penalises_one_bad_proxy():
    """AZ-NAS Eq.(12)의 성질: 한 proxy에서만 최상위인 후보보다 고르게 좋은 후보를 고른다."""
    cal = A.Calibrator(np.random.default_rng(0).standard_normal((5000, 3)))
    consistent = np.array([[1.5, 1.5, 1.5]])
    spiky = np.array([[9.0, -1.0, -1.0]])  # 평균이 consistent보다 높도록
    zc, uc = cal.transform(consistent)
    zs, us = cal.transform(spiky)

    # 합의형은 고르게 좋은 쪽을 고른다
    for agg in (A.AZNASNonLinear(n_calib=5000), A.SoftConsensus(p=4.0), A.WorstRank()):
        assert agg(zc, uc)[0] > agg(zs, us)[0], agg.name

    # z 공간 선형합과 spike 합은 뾰족한 후보에 속는다 — 우리 가설의 핵심 대비
    for agg in (A.LinearWeighted(), A.SpikeRankSum()):
        assert agg(zs, us)[0] > agg(zc, uc)[0], agg.name


def test_aznas_nonlinear_penalises_more_than_linear():
    """AZ-NAS Table 4의 NL > L 우위가 구조적으로 재현되는지 (상대 마진 비교)."""
    cal = A.Calibrator(np.random.default_rng(0).standard_normal((5000, 3)))
    zc, uc = cal.transform(np.array([[1.5, 1.5, 1.5]]))
    zs, us = cal.transform(np.array([[9.0, -1.0, -1.0]]))

    def rel_margin(agg):
        good, bad = agg(zc, uc)[0], agg(zs, us)[0]
        return (good - bad) / (abs(good) + abs(bad) + 1e-12)

    assert rel_margin(A.AZNASNonLinear(n_calib=5000)) > rel_margin(A.AZNASLinear(n_calib=5000))


def test_ascending_percentile_bounds():
    cal = A.Calibrator(np.random.default_rng(1).standard_normal((1000, 2)))
    _, u = cal.transform(np.array([[5.0, -5.0]]))
    g = A._ascending_percentile(u, cal.n_calib)
    assert 0.0 < g.min() <= g.max() <= 1.0
    assert g[0, 0] > g[0, 1]  # 높은 점수가 높은 백분위
