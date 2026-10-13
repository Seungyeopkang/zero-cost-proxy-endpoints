"""EXP05 — 끝점 곡선 (모든 예산 B에 대해 전수 계산).

exp04는 B를 8점만 봤다. 안전 예산(어느 B부터 무작위보다 나빠지지 않는가)을 말하려면
격자가 아니라 곡선이 필요하다. 전수 벤치마크라 정확히 계산된다:

    endpoint(B) = 점수 내림차순으로 정렬한 뒤 acc의 누적 최대값
    random(B)   = E[max of B] = sum_i acc_(i) * C(i-1,B-1)/C(n,B)   (표본추출 없음)

시드도 오차도 없다. 산출: 모든 scorer x dataset x B=1..B_MAX 의 delta 곡선.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas.aggregate import (  # noqa: E402
    AZNASLinear, AZNASNonLinear, Calibrator, LinearWeighted, MeanRank,
    SoftConsensus, WorstRank,
)
from exnas.benchmark import NB201Benchmark  # noqa: E402
from exnas.logging_utils import config_hash, run_header  # noqa: E402
from exnas.space import all_archs  # noqa: E402

B_MAX = 1000


def random_baseline_curve(acc: np.ndarray, b_max: int) -> np.ndarray:
    """E[max of B], B=1..b_max. 복원 없이 추출. 정확 계산."""
    n = len(acc)
    a = np.sort(acc)
    i = np.arange(1, n + 1, dtype=np.float64)
    lg_i = gammaln(i)
    out = np.empty(b_max)
    for b in range(1, b_max + 1):
        logw = (lg_i - gammaln(float(b)) - gammaln(i - b + 1)
                - (gammaln(n + 1) - gammaln(b + 1) - gammaln(n - b + 1)))
        w = np.where(i >= b, np.exp(logw), 0.0)
        out[b - 1] = float(np.dot(a, w / w.sum()))
    return out


def safe_budget(delta: np.ndarray) -> int | None:
    """delta(B) >= 0 이 이후 모든 B에서 유지되는 최소 B. 없으면 None."""
    ok = delta >= 0
    if not ok.any():
        return None
    # 뒤에서부터 최초로 False가 나오는 지점 다음
    bad = np.where(~ok)[0]
    return int(bad[-1] + 2) if len(bad) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/zc_nasbench201.json")
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--b-max", type=int, default=B_MAX)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    config = vars(args) | {"experiment": "exp05_endpoint_curve"}
    h = config_hash(config)
    out = Path(args.out) / f"exp05_{h}.npz"
    meta_path = Path(args.out) / f"exp05_{h}.json"

    archs = list(all_archs())
    store: dict[str, np.ndarray] = {}
    meta: list[dict] = []

    for ds in NB201Benchmark.DATASETS:
        bench = NB201Benchmark(ds, args.data)
        acc = bench.all_acc()
        scores = bench.proxy_scores(archs)
        cal = Calibrator(scores[np.random.default_rng(0).choice(
            len(archs), args.n_calib, replace=False)])
        z, u = cal.transform(scores)

        scorers: dict[str, np.ndarray] = {p: scores[:, j] for j, p in enumerate(bench.proxy_names)}
        for agg in (AZNASNonLinear(args.n_calib), AZNASLinear(args.n_calib),
                    LinearWeighted(), MeanRank(), SoftConsensus(p=4.0), WorstRank()):
            scorers[agg.name] = agg(z, u)

        base = random_baseline_curve(acc, args.b_max)
        store[f"{ds}/random"] = base
        top10 = np.argsort(-acc)[: len(acc) // 10]

        print(f"\n=== {ds} ===")
        print(f"{'scorer':22s}{'전역ρ':>8}{'상위10%ρ':>10}{'최악 delta':>12}{'@B':>6}{'안전예산':>9}")
        for name, s in scorers.items():
            end = np.maximum.accumulate(acc[np.argsort(-s)])[: args.b_max]
            delta = end - base
            store[f"{ds}/{name}"] = delta
            g = float(spearmanr(s, acc).statistic)
            t = float(spearmanr(s[top10], acc[top10]).statistic)
            sb = safe_budget(delta)
            worst = float(delta.min()); wb = int(delta.argmin()) + 1
            meta.append({"dataset": ds, "scorer": name, "global_rho": g, "top10_rho": t,
                         "worst_delta": worst, "worst_b": wb, "safe_budget": sb,
                         "delta_at": {b: float(delta[b - 1]) for b in (5, 20, 100, 500)}})
            print(f"{name:22s}{g:8.3f}{t:10.3f}{worst:12.2f}{wb:6d}"
                  f"{(str(sb) if sb else '>%d' % args.b_max):>9}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, b_max=args.b_max, **store)
    meta_path.write_text(json.dumps({"header": run_header(config), "rows": meta},
                                    ensure_ascii=False, indent=1))
    print(f"\nwrote {out}  and  {meta_path}")


if __name__ == "__main__":
    main()
