"""고정 선택예산 파이프라인 — 이 연구의 측정 대상.

    후보 N개 생성 -> proxy로 점수 -> 상위 B개를 선택 -> 벤치마크에서 최고 정확도를 반환

N = B 이면 필터링이 없고, N -> 무한이면 proxy의 전역 argmax로 수렴한다.
양 극단이 모두 나쁘면 중간에 최적점 N*가 존재한다. 그 곡선을 그리는 게 목적.

이 구현은 학습된 네트워크를 실행하지 않고 사전 계산된 벤치마크 표를 조회한다.
학습예산 B는 실제로 학습했다고 가정하는 서로 다른 후보 구조의 수다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from . import metrics
from .aggregate import Aggregator, Calibrator
from .search import make_fitness, random_pool, regularized_evolution
from .space import Arch

Mode = Literal["random", "ea"]


@dataclass
class RunResult:
    mode: str
    aggregator: str
    n_candidates: int
    train_budget: int
    seed: int
    # --- 1순위 지표
    best_acc: float
    regret: float
    # --- 진단 (벤치마크 테이블이므로 계산 가능. 방법의 일부가 아니다)
    population_kl: float
    pool_unique: int
    pool_best_acc: float
    selection_gap: float
    oracle_best_acc: float
    pool_tau: float
    pool_top10_tau: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_ordered(archs: Sequence[Arch]) -> list[Arch]:
    seen: set[Arch] = set()
    out: list[Arch] = []
    for a in archs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def make_calibrator(bench, rng: np.random.Generator, n_calib: int = 2000) -> Calibrator:
    """탐색공간에서 무작위 표본을 뽑아 proxy 분포 기준을 고정한다 (학습예산 미소모)."""
    sample = random_pool(n_calib, rng)
    return Calibrator(bench.proxy_scores(sample))


def run_pipeline(
    bench,
    aggregator: Aggregator,
    calibrator: Calibrator,
    mode: Mode,
    n_candidates: int,
    train_budget: int,
    seed: int,
    population_size: int = 50,
    sample_size: int = 10,
) -> RunResult:
    if train_budget > n_candidates:
        raise ValueError("train_budget must be <= n_candidates")
    rng = np.random.default_rng(seed)

    if mode == "random":
        history = random_pool(n_candidates, rng)
    elif mode == "ea":
        fitness = make_fitness(bench.proxy_scores, calibrator, aggregator, run_seed=seed)
        history = regularized_evolution(
            n_candidates,
            fitness,
            rng,
            population_size=min(population_size, n_candidates),
            sample_size=min(sample_size, n_candidates),
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    pool = _unique_ordered(history)
    scores = bench.proxy_scores(pool)
    z, u = calibrator.transform(scores)
    if hasattr(aggregator, "score_archs"):
        agg = aggregator.score_archs(pool, seed)
    else:
        agg = aggregator(z, u)

    b = min(train_budget, len(pool))
    selected_idx = np.argsort(-agg)[:b]          # <- 여기까지만 proxy를 본다
    selected_acc = bench.true_acc([pool[int(i)] for i in selected_idx])  # 사전 계산 표 조회

    pool_acc = bench.true_acc(pool)              # 진단 전용
    best_acc = float(selected_acc.max())

    return RunResult(
        mode=mode,
        aggregator=aggregator.name,
        n_candidates=n_candidates,
        train_budget=train_budget,
        seed=seed,
        best_acc=best_acc,
        regret=metrics.regret(best_acc, bench.best_acc),
        population_kl=metrics.population_kl(history),
        pool_unique=len(pool),
        pool_best_acc=float(pool_acc.max()),
        selection_gap=metrics.selection_gap(pool_acc, selected_acc),
        oracle_best_acc=metrics.oracle_best_of(pool_acc, b),
        pool_tau=metrics.tau(agg, pool_acc),
        pool_top10_tau=metrics.top_frac_tau(agg, pool_acc, 0.1),
    )
