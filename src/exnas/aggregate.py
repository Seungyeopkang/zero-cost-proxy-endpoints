"""proxy 결합 방식.

가설: 결합 방식이 안전 후보 수 N*를 미는 정도가 다르다.
    단일 proxy  <  선형 가중합(RoBoT/DAEP)  <  비선형 합의(AZ-NAS)
선형합은 한 proxy만 크게 나와도 통과하지만, 합의 기반은 모든 proxy를 동시에 속여야 하기 때문.

설계 주의 — 순위 기반 결합은 '어떤 집합에서 순위를 매기는가'에 의존한다.
EA 도중에는 평가 집합이 계속 커지므로, 순위를 그때그때 다시 매기면
같은 아키텍처의 fitness가 시간에 따라 변해 탐색이 오염된다.
따라서 탐색공간에서 한 번 뽑은 **calibration pool**로 proxy별 경험적 CDF를 고정하고,
이후 모든 아키텍처를 그 CDF에 투영해 정규화 순위 u를 얻는다.
calibration pool은 proxy 점수만 쓰므로 학습예산을 소모하지 않는다.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .space import Arch, SPACE_SIZE, arch_index


class Calibrator:
    """proxy별 z-정규화와 정규화 순위(u)를 고정된 기준으로 제공한다.

    u in (0, 1], 작을수록 상위. u = (그 점수 이상인 calibration 표본의 비율).
    """

    def __init__(self, calib_scores: np.ndarray) -> None:
        if calib_scores.ndim != 2:
            raise ValueError("calib_scores must be (n_calib, n_proxies)")
        self.n_calib, self.n_proxies = calib_scores.shape
        self.mu = calib_scores.mean(axis=0)
        sd = calib_scores.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.sorted = np.sort(calib_scores, axis=0)  # 오름차순

    def transform(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (scores - self.mu) / self.sd
        u = np.empty_like(scores, dtype=np.float64)
        for k in range(self.n_proxies):
            n_below = np.searchsorted(self.sorted[:, k], scores[:, k], side="left")
            u[:, k] = 1.0 - n_below / self.n_calib
        # 0 방지 (log 사용). 최상위는 1/n_calib로 고정.
        np.clip(u, 1.0 / self.n_calib, 1.0, out=u)
        return z, u


class Aggregator(Protocol):
    name: str

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """(n,K) z-점수와 정규화 순위 -> (n,) 결합 점수. 클수록 유망."""


@dataclass
class SingleProxy:
    """proxy 하나만 사용 (K=1 베이스라인)."""

    index: int = 0

    @property
    def name(self) -> str:
        return f"single[{self.index}]"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return z[:, self.index]


@dataclass
class LinearWeighted:
    """가중 선형합. RoBoT / DAEP 계열. 한 proxy만 커도 통과한다."""

    weights: np.ndarray | None = None
    name: str = "linear"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        w = np.ones(z.shape[1]) if self.weights is None else np.asarray(self.weights, dtype=float)
        return z @ (w / w.sum())


@dataclass
class MeanRank:
    """평균 순위. 순위 공간에서의 선형 결합."""

    name: str = "mean_rank"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return -u.mean(axis=1)


@dataclass
class SpikeRankSum:
    """의도적으로 착취 가능한 컨트롤: sum(-log u), u = 상위 비율(작을수록 상위).

    한 proxy에서 극단적 상위면 -log u가 커서 나머지가 하위여도 이긴다.
    AZ-NAS와 부호가 반대인 형태이며, '합의처럼 보이지만 아닌' 결합의 대조군으로 남긴다.
    tests/test_aggregate.py가 이 성질을 고정한다.
    """

    name: str = "spike_rank_sum"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return (-np.log(u)).sum(axis=1)


def _ascending_percentile(u: np.ndarray, n_calib: int) -> np.ndarray:
    """u(상위 비율, 작을수록 상위) -> g(오름차순 순위 백분위, 1이 최고)."""
    return np.clip(1.0 - u + 1.0 / n_calib, 1.0 / n_calib, 1.0)


@dataclass
class AZNASNonLinear:
    """AZ-NAS Eq.(12) 비선형 rank aggregation (Lee & Ham, CVPR 2024).

        s_AZ(i) = sum_M log( Rank(s_M(i)) / m )

    Rank(.)는 오름차순이므로 값이 클수록 좋고, /m으로 (0,1] 백분위가 된다.
    원문: "It penalizes the final score more severely when one of the proxies
    indicates subpar performance, allowing us to find a network highly-ranked
    across all the proxies evenly."

    주의: /m은 모든 후보에 같은 상수 -K*log(m)을 더하므로 고정 풀 안의 argmax를 바꾸지 않는다.
    풀 크기 의존성은 Rank(.)를 어느 집합에서 매기느냐를 통해서만 들어온다 — 우리 실험의 핵심 축.
    여기서는 Calibrator의 고정 기준으로 백분위를 매긴다.
    """

    n_calib: int = 2000
    name: str = "aznas_nl"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        g = _ascending_percentile(u, self.n_calib)
        return np.log(g).sum(axis=1)


@dataclass
class AZNASLinear:
    """AZ-NAS Table 4의 선형(L) rank aggregation 비교군: sum_M Rank(s_M)/m.

    순위 공간이라 한 proxy의 기여가 1로 상한이 걸린다 (z 공간 선형합과 다른 점).
    """

    n_calib: int = 2000
    name: str = "aznas_l"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return _ascending_percentile(u, self.n_calib).sum(axis=1)


@dataclass
class SoftConsensus:
    """멱평균 기반 합의 결합 — 최악 순위에 가중치를 준다.

    score = -( mean(u^p) )^(1/p)
    p=1이면 MeanRank, p->무한이면 WorstRank. p>1이면 하위 순위 하나가 전체를 끌어내린다.
    AZ-NAS의 log-rank 합과 다른 형태로 같은 성질을 내는 대안이며, p로 강도를 연속 조절한다.
    """

    p: float = 4.0

    @property
    def name(self) -> str:
        return f"soft_consensus[p={self.p:g}]"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return -np.power(np.mean(np.power(u, self.p), axis=1), 1.0 / self.p)


@dataclass
class WorstRank:
    """최악 순위 기준 (완전 합의). 착취 저항의 상한 참조점."""

    name: str = "worst_rank"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return -u.max(axis=1)


@dataclass
class AntiOracle:
    """컨트롤: 부호를 뒤집는다. 무작위보다 나쁘게 나와야 정상."""

    base: Aggregator

    @property
    def name(self) -> str:
        return f"anti[{self.base.name}]"

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return -self.base(z, u)


@dataclass
class RandomProxy:
    """컨트롤: 구조별 고정 표준정규 무작위 점수. 필터링의 하한선.

    점수는 구조 인덱스와 실행 시드로 인덱싱되는 난수장이다. 따라서 같은 구조는
    한 실행의 진화 중과 최종 선택에서 같은 점수를 받고, 서로 다른 실행 시드는
    독립적인 난수장을 사용한다.
    """

    seed: int = 0
    name: str = "random_proxy"
    _cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _field(self, run_seed: int) -> np.ndarray:
        run_seed = int(run_seed)
        if run_seed not in self._cache:
            rng = np.random.default_rng(np.random.SeedSequence([int(self.seed), run_seed]))
            self._cache[run_seed] = rng.standard_normal(SPACE_SIZE)
        return self._cache[run_seed]

    def score_arch(self, arch: Arch, run_seed: int) -> float:
        """Return this run's fixed random score for one architecture."""
        return float(self._field(run_seed)[arch_index(arch)])

    def score_archs(self, archs: list[Arch], run_seed: int) -> np.ndarray:
        """Return architecture-keyed scores independent of input order or batching."""
        scores = self._field(run_seed)
        return np.fromiter((scores[arch_index(arch)] for arch in archs), dtype=float,
                           count=len(archs))

    def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vector-only compatibility path when architecture IDs are unavailable.

        The search pipeline does not use this positional fallback. It dispatches
        through score_arch/score_archs so candidate fitness remains architecture-keyed.
        """
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal(z.shape[0])


def subset(agg_cls, indices: list[int]):
    """앙상블 크기 K 스윕용 — proxy 부분집합만 보는 결합기로 감싼다."""

    inner = agg_cls()

    @dataclass
    class _Subset:
        name: str = f"{inner.name}[K={len(indices)}]"

        def __call__(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
            return inner(z[:, indices], u[:, indices])

    return _Subset()


def default_aggregators(seed: int = 0) -> list[Aggregator]:
    """기본 세트: 단일 / 선형 / 순위평균 / 합의 / 최악순위 + 컨트롤 2종."""
    return [
        SingleProxy(0),
        LinearWeighted(),
        MeanRank(),
        AZNASLinear(),
        AZNASNonLinear(),
        SoftConsensus(p=4.0),
        WorstRank(),
        SpikeRankSum(),
        AntiOracle(SingleProxy(0)),
        RandomProxy(seed=seed),
    ]
