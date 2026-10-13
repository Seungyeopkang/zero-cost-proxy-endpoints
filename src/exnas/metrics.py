"""측정량.

핵심은 population_kl — '최적화를 얼마나 세게 했는가'의 단일 축이다.
후보 수 N을 늘리는 것과 EA 세대를 늘리는 것을 같은 축에 올리기 위해 쓴다.
(RLHF의 reward model overoptimization 곡선에서 KL이 하는 역할과 같다.
 TODO(Q6): Gao et al.의 KL 정의를 원문 확인 후 best-of-n 대응 항을 정확히 맞출 것.)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kendalltau

from .space import N_EDGES, N_OPS, Arch, edge_op_matrix


def population_kl(archs: Sequence[Arch], eps: float = 1e-12) -> float:
    """KL( 개체군 분포 || 균등 분포 ), edge별 범주형 분포의 합.

    균등 무작위 표본이면 0에 가깝고, 특정 op으로 수렴할수록 커진다.
    최대값은 N_EDGES * log(N_OPS).
    """
    if len(archs) == 0:
        return 0.0
    ops = edge_op_matrix(archs)
    total = 0.0
    for e in range(N_EDGES):
        counts = np.bincount(ops[:, e], minlength=N_OPS).astype(np.float64)
        p = counts / counts.sum()
        nz = p > eps
        total += float(np.sum(p[nz] * np.log(p[nz] * N_OPS)))
    return total


MAX_KL = N_EDGES * float(np.log(N_OPS))


def tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b. 표본이 2개 미만이거나 상수면 nan."""
    if len(a) < 2:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def top_frac_tau(scores: np.ndarray, acc: np.ndarray, frac: float = 0.1) -> float:
    """진짜 성능 상위 frac 구간으로 제한한 tau. 상위구간 붕괴 측정용."""
    n = max(2, int(round(len(acc) * frac)))
    top = np.argsort(-acc)[:n]
    return tau(scores[top], acc[top])


def regret(found_acc: float, best_acc: float) -> float:
    """탐색 공간 최적 대비 손실 (퍼센트 포인트)."""
    return float(best_acc - found_acc)


def selection_gap(pool_acc: np.ndarray, selected_acc: np.ndarray) -> float:
    """proxy 선택의 손해 — 풀 안의 진짜 최고와 실제로 고른 것의 차이.

    proxy가 완벽하면 0. 클수록 착취당했다는 뜻.
    """
    if len(pool_acc) == 0 or len(selected_acc) == 0:
        return float("nan")
    return float(pool_acc.max() - selected_acc.max())


def oracle_best_of(pool_acc: np.ndarray, budget: int) -> float:
    """같은 풀에서 진짜 acc를 안다고 가정했을 때의 상한 (proxy 무용지물 기준선)."""
    if len(pool_acc) == 0:
        return float("nan")
    return float(np.sort(pool_acc)[-budget:].max())
