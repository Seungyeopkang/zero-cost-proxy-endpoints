"""Synthetic and NAS-Bench-201 benchmark adapters for the experiments.

Benchmark accuracy is used only to evaluate selected architectures. Candidate
generation and proxy fitness must not read it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .space import SPACE_SIZE, Arch, N_EDGES, N_OPS, arch_index, arch_to_str


@runtime_checkable
class Benchmark(Protocol):
    name: str
    proxy_names: tuple[str, ...]

    def true_acc(self, archs: Sequence[Arch]) -> np.ndarray:
        """Evaluation-only benchmark accuracy; search and proxy fitness must not read it."""

    def proxy_scores(self, archs: Sequence[Arch]) -> np.ndarray:
        """(len(archs), n_proxies) proxy 점수 행렬. 클수록 좋다고 가정."""

    @property
    def best_acc(self) -> float:
        """regret 계산 기준값 (탐색 공간 전체 최적)."""


def _features(idx: np.ndarray) -> np.ndarray:
    """arch index 배열 -> (n, N_OPS+1) 구조 특징.

    앞 N_OPS열: 각 op이 몇 번 쓰였는지. 마지막 열: conv 가중합(파라미터 수 대용).
    """
    n = len(idx)
    ops = np.empty((n, N_EDGES), dtype=np.int64)
    rest = idx.copy()
    for e in range(N_EDGES - 1, -1, -1):
        rest, ops[:, e] = np.divmod(rest, N_OPS)
    counts = np.zeros((n, N_OPS), dtype=np.float64)
    for op in range(N_OPS):
        counts[:, op] = (ops == op).sum(axis=1)
    # nor_conv_3x3(=3)이 파라미터를 가장 많이 쓰고 nor_conv_1x1(=2)이 그다음
    params_like = 3.0 * counts[:, 3] + 1.0 * counts[:, 2]
    return np.concatenate([counts, params_like[:, None]], axis=1)


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


class SyntheticBenchmark:
    """제어 가능한 합성 벤치마크.

    proxy_k = a_k * z(true_acc)
              + shared_bias * z(params_like)          <- 모든 proxy가 공유하는 편향
              + private_bias * z(u_k . features)      <- proxy마다 다른 편향
              + noise * eps

    shared_bias를 키우면 proxy 오차가 서로 상관되어 앙상블이 무력해진다.
    private_bias를 키우면 오차가 decorrelated되어 앙상블이 강해진다.
    이 두 손잡이로 kill switch를 사전 등록 가능한 형태로 검증한다.

    ⚠️ 합성 데이터 결과는 그 자체로 주장의 근거가 아니다.
       실데이터(NAS-Bench-201)에서 재현된 뒤에만 논문 주장에 쓴다.
    """

    name = "synthetic"

    def __init__(
        self,
        n_proxies: int = 8,
        seed: int = 0,
        signal: float = 1.0,
        shared_bias: float = 0.8,
        private_bias: float = 0.6,
        noise: float = 0.4,
    ) -> None:
        rng = np.random.default_rng(seed)
        idx = np.arange(SPACE_SIZE)
        feats = _features(idx)

        # --- ground truth: conv이 좋고 none이 나쁘다 + 비선형 상호작용 + 소량 잡음
        w_true = np.array([-2.5, 0.3, 1.0, 1.6, 0.2, 0.0])
        base = feats @ w_true
        interaction = -0.25 * (feats[:, 3] ** 2)  # 무한정 키운다고 좋아지진 않음
        acc = base + interaction + 0.35 * rng.standard_normal(SPACE_SIZE)
        # 60~95% 범위로 스케일 (NB-201 CIFAR-10 대역과 비슷하게)
        acc = _z(acc)
        self._acc = 87.0 + 4.0 * acc
        self._acc = np.clip(self._acc, 20.0, 95.0)

        # --- proxies
        z_acc = _z(self._acc)
        z_params = _z(feats[:, -1])
        z_feats = np.apply_along_axis(_z, 0, feats)
        directions = rng.standard_normal((n_proxies, feats.shape[1]))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        scores = np.empty((SPACE_SIZE, n_proxies))
        for k in range(n_proxies):
            private = _z(z_feats @ directions[k])
            scores[:, k] = (
                signal * z_acc
                + shared_bias * z_params
                + private_bias * private
                + noise * rng.standard_normal(SPACE_SIZE)
            )
        self._scores = scores
        self.proxy_names = tuple(f"synthetic_p{k}" for k in range(n_proxies))
        self.config = {
            "n_proxies": n_proxies,
            "seed": seed,
            "signal": signal,
            "shared_bias": shared_bias,
            "private_bias": private_bias,
            "noise": noise,
        }

    def _idx(self, archs: Sequence[Arch]) -> np.ndarray:
        return np.fromiter((arch_index(a) for a in archs), dtype=np.int64, count=len(archs))

    def true_acc(self, archs: Sequence[Arch]) -> np.ndarray:
        return self._acc[self._idx(archs)]

    def proxy_scores(self, archs: Sequence[Arch]) -> np.ndarray:
        return self._scores[self._idx(archs)]

    @property
    def best_acc(self) -> float:
        return float(self._acc.max())


class NB201Benchmark:
    """NAS-Bench-201 실데이터 백엔드 (NAS-Bench-Suite-Zero 사전계산 proxy 점수).

    데이터: `data/zc_nasbench201.json` (44.3MB, Apache 2.0)
        gdown 1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H
    커버리지: 15,625 아키텍처 전수 x {cifar10, cifar100, ImageNet16-120} x 13 proxy.
    (ninapro / svhn / scifar100은 199개뿐이라 이 연구에서 쓰지 않는다.)

    목표 정확도는 NBSZ가 제공하는 **validation accuracy**다. test accuracy는 이 파일에 없다.
    탐색 신호로는 validation이 오히려 올바르므로(test를 탐색에 쓰면 안 됨) 그대로 사용하고,
    논문에서 'NAS-Bench-201 validation accuracy'라고 명시한다.

    proxy 부호: 원본 그대로 쓴다. `plain`은 전역 Spearman이 -0.267로 음의 상관이지만
    부호를 교정하려면 정답 라벨을 봐야 하므로(누수) 건드리지 않는다.
    앙상블에 해로운 멤버가 섞이는 것은 실제 상황이며, K 스윕의 부분집합 분산으로 관측된다.
    """

    name = "nb201"
    DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
    PROXIES = (
        "epe_nas", "fisher", "flops", "grad_norm", "grasp", "jacov", "l2_norm",
        "nwot", "params", "plain", "snip", "synflow", "zen",
    )

    def __init__(
        self,
        dataset: str = "cifar10",
        path: str | Path = "data/zc_nasbench201.json",
        proxies: Sequence[str] | None = None,
    ) -> None:
        if dataset not in self.DATASETS:
            raise ValueError(f"dataset must be one of {self.DATASETS}, got {dataset!r}")
        self.dataset = dataset
        self.proxy_names = tuple(proxies) if proxies else self.PROXIES

        with open(path) as f:
            raw = json.load(f)[dataset]
        if len(raw) != SPACE_SIZE:
            raise ValueError(f"expected {SPACE_SIZE} architectures, found {len(raw)}")

        acc = np.full(SPACE_SIZE, np.nan)
        scores = np.full((SPACE_SIZE, len(self.proxy_names)), np.nan)
        for key, rec in raw.items():
            idx = arch_index(tuple(int(t) for t in key.strip("()").split(",")))
            acc[idx] = rec["val_accuracy"]
            for j, pr in enumerate(self.proxy_names):
                scores[idx, j] = rec[pr]["score"]
        if np.isnan(acc).any() or np.isnan(scores).any():
            raise ValueError("데이터에 결측이 있다 — 인덱싱 또는 파일 확인 필요")

        self._acc = acc
        self._scores = scores
        self.config = {
            "backend": "nb201", "dataset": dataset,
            "proxies": list(self.proxy_names), "objective": "val_accuracy",
        }

    def _idx(self, archs: Sequence[Arch]) -> np.ndarray:
        return np.fromiter((arch_index(a) for a in archs), dtype=np.int64, count=len(archs))

    def true_acc(self, archs: Sequence[Arch]) -> np.ndarray:
        return self._acc[self._idx(archs)]

    def proxy_scores(self, archs: Sequence[Arch]) -> np.ndarray:
        return self._scores[self._idx(archs)]

    @property
    def best_acc(self) -> float:
        return float(self._acc.max())

    def all_acc(self) -> np.ndarray:
        """진단 전용 — 탐색은 호출하지 않는다."""
        return self._acc
