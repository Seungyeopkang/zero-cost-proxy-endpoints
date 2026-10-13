"""탐색 알고리즘 — 후보를 생성하는 부분.

두 방식 모두 'proxy 평가 N회'를 소비하고 평가 이력을 반환한다.
학습예산 B는 여기서 쓰지 않는다 (pipeline이 담당). 따라서 두 방식은 같은 축에서 비교된다.

  random_pool  : 균등 무작위. 최적화 강도 최소 (KL ~ 0)
  regularized_evolution : aging evolution. proxy 점수를 fitness로 쓴다 -> KL 증가

핵심 대조: 같은 N에서 EA는 KL이 크므로 착취가 더 심해야 한다.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Sequence

import numpy as np

from .space import Arch, mutate, random_arch

FitnessFn = Callable[[Arch], float]


def random_pool(n: int, rng: np.random.Generator) -> list[Arch]:
    """균등 무작위로 n개 생성 (중복 허용 — EA와 평가 횟수를 맞추기 위함)."""
    return [random_arch(rng) for _ in range(n)]


def regularized_evolution(
    n: int,
    fitness: FitnessFn,
    rng: np.random.Generator,
    population_size: int = 50,
    sample_size: int = 10,
) -> list[Arch]:
    """Aging evolution (Real et al., AAAI 2019). proxy 평가 n회를 소비한다.

    sample_size가 선택압을 조절한다 (클수록 강한 선택 -> KL 증가).
    반환값은 평가된 순서대로의 전체 이력.
    초기 population_size개 무작위 초기화 평가도 총 n회에 포함한다.
    """
    if n < population_size:
        raise ValueError(f"n({n}) must be >= population_size({population_size})")
    if sample_size > population_size:
        raise ValueError("sample_size must be <= population_size")

    history: list[Arch] = []
    population: deque[tuple[Arch, float]] = deque(maxlen=population_size)

    while len(population) < population_size:
        arch = random_arch(rng)
        population.append((arch, fitness(arch)))
        history.append(arch)

    while len(history) < n:
        idx = rng.choice(len(population), size=sample_size, replace=False)
        parent = max((population[int(i)] for i in idx), key=lambda t: t[1])[0]
        child = mutate(parent, rng)
        population.append((child, fitness(child)))
        history.append(child)

    return history


def make_fitness(
    proxy_lookup: Callable[[Sequence[Arch]], np.ndarray],
    calibrator,
    aggregator,
    run_seed: int = 0,
) -> FitnessFn:
    """단일 아키텍처를 결합 점수 하나로 매핑하는 fitness 함수를 만든다.

    calibrator가 고정 기준을 제공하므로, 같은 아키텍처는 항상 같은 fitness를 받는다.
    """

    if hasattr(aggregator, "score_arch"):
        return lambda arch: float(aggregator.score_arch(arch, run_seed))

    def fn(arch: Arch) -> float:
        scores = proxy_lookup([arch])
        z, u = calibrator.transform(scores)
        return float(aggregator(z, u)[0])

    return fn
