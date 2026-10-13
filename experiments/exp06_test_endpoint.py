"""EXP06 — 테스트 정확도 기준 끝점 곡선 (리뷰 P0-1 대응).

배포 지향 절차를 그대로 구현한다.
    1. proxy 점수 상위 B개를 학습한다.
    2. 그 B개 중 **검증 정확도**가 가장 높은 구조를 고른다.
    3. 그 구조의 **테스트 정확도**를 보고한다.

무작위 기준선도 같은 절차로 정의된다. 크기 B의 균등 부분집합 S에 대하여
E[test(argmax_val S)] 이며, 검증 정확도 오름차순 순위를 i라 하면 정확히

    E_B = Σ_i test(σ(i)) · C(i-1, B-1) / C(n, B)

이다. σ 는 검증 정확도 오름차순 정렬이고, 표본추출이 없으므로 시드도 오차도 없다.
무작위 선택이 이길 확률도 같은 전개로 정확히 계산된다.

    P(test(argmax_val S) > e) = Σ_{i: test(σ(i)) > e} C(i-1, B-1) / C(n, B)

검증 정확도 동점은 argmax 를 모호하게 만들므로, 동점 안에서 테스트가 높은 쪽/낮은 쪽을
고르는 두 극단(optimistic / pessimistic)을 모두 계산한다. 프록시 점수 동점은 벤치마크의
고정 구조 열거 순서로 해소하며, 표의 비교는 프록시 상한에서 무작위 기준선 하한을 뺀다.

  python3 experiments/exp06_test_endpoint.py --out results/exp06_test_endpoint_favbounds
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
from exnas.space import all_archs  # noqa: E402

B_MAX = 1000
ACC_KEY = {"cifar10": ("c10_valid", "c10_test"),
           "cifar100": ("c100_valid", "c100_test"),
           "ImageNet16-120": ("in16_valid", "in16_test")}


def logC(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def order_stat_weights(n: int, b_max: int) -> np.ndarray:
    """W[b-1, i] = C(i, b-1) / C(n, b)  (i는 0-기반 오름차순 순위)."""
    i = np.arange(n, dtype=np.float64)
    W = np.zeros((b_max, n))
    for b in range(1, b_max + 1):
        ok = i >= b - 1
        lw = np.where(ok, logC(i, b - 1) - logC(n, b), -np.inf)
        W[b - 1] = np.exp(lw)
    return W


def test_endpoint_bounds(order: np.ndarray, val: np.ndarray,
                         test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 prefix의 최고 validation 구조들 중 test 최댓값/최솟값을 반환한다."""
    v = val[order][:B_MAX]
    t = test[order][:B_MAX]
    upper = np.empty(B_MAX)
    lower = np.empty(B_MAX)
    best_val = -np.inf
    best_test_max = -np.inf
    best_test_min = np.inf
    for b, (v_new, t_new) in enumerate(zip(v, t)):
        if v_new > best_val:
            best_val = v_new
            best_test_max = best_test_min = t_new
        elif v_new == best_val:
            best_test_max = max(best_test_max, t_new)
            best_test_min = min(best_test_min, t_new)
        upper[b] = best_test_max
        lower[b] = best_test_min
    return upper, lower


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/zc_nasbench201.json")
    ap.add_argument("--acc", default="data/nb201_acc.npz")
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--out", default="results/exp06_test_endpoint_favbounds")
    args = ap.parse_args()

    acc_tab = np.load(args.acc)
    archs = list(all_archs())
    n = len(archs)
    W = order_stat_weights(n, B_MAX)

    store, meta = {}, []
    for ds in NB201Benchmark.DATASETS:
        bench = NB201Benchmark(ds, args.data)
        nbsz_val = bench.all_acc()
        vkey, tkey = ACC_KEY[ds]
        val, test = acc_tab[vkey], acc_tab[tkey]
        agree = float(np.abs(val - nbsz_val).max())
        rho = float(spearmanr(val, nbsz_val).statistic)
        print(f"\n=== {ds}  API val vs NBSZ val: max|diff| {agree:.4f}, spearman {rho:.6f}")
        print(f"    val/test spearman {float(spearmanr(val, test).statistic):.4f}, "
              f"unique val {len(np.unique(val))}, max val-tie {int(np.bincount(np.unique(val, return_inverse=True)[1]).max())}")

        # 무작위 기준선 (validation 동점에서 test 상한/하한)
        base, sigma_of = {}, {}
        for tag, second in (("upper", test), ("lower", -test)):
            # 1차 키 val 오름차순, 동점 안에서는 second 오름차순.
            # 정렬의 마지막 원소가 선택되므로 upper는 test 큰 쪽, lower는 작은 쪽을 둔다.
            sigma = np.lexsort((second, val))
            sigma_of[tag] = sigma
            base[tag] = W @ test[sigma]
        store[f"{ds}/random_test_upper"] = base["upper"]
        store[f"{ds}/random_test_lower"] = base["lower"]

        scores = {p: bench.proxy_scores(archs)[:, j] for j, p in enumerate(bench.proxy_names)}
        raw = bench.proxy_scores(archs)
        cal = Calibrator(raw[np.random.default_rng(0).choice(n, args.n_calib, replace=False)])
        z, u = cal.transform(raw)
        for agg in (LinearWeighted(), AZNASNonLinear(args.n_calib), AZNASLinear(args.n_calib),
                    MeanRank(), SoftConsensus(p=4.0), WorstRank()):
            scores[agg.name] = agg(z, u)

        print(f"{'scorer':24s}{'d20':>8}{'worst':>9}{'@B':>6}{'P@worst':>10}{'unsafe':>8}")
        for name, s in scores.items():
            # Proxy-score ties use the stable NAS-Bench-201 architecture enumeration order.
            order = np.lexsort((np.arange(n), -s))
            proxy_upper, proxy_lower = test_endpoint_bounds(order, val, test)
            # Favor the proxy: its test-accuracy upper endpoint vs the random lower endpoint.
            delta_fav = proxy_upper - base["lower"]
            wb = int(delta_fav.argmin()) + 1
            e = float(proxy_upper[wb - 1])
            # The random comparator uses the lower test tie policy for the same event.
            random_lower_order = sigma_of["lower"]
            pw = float(W[wb - 1][test[random_lower_order] > e].sum())
            unsafe = int((delta_fav < -1.0).sum())
            store[f"{ds}/{name}/proxy_upper_minus_random_lower"] = delta_fav
            store[f"{ds}/{name}/proxy_lower_minus_random_upper"] = proxy_lower - base["upper"]
            meta.append(dict(dataset=ds, scorer=name,
                             d20=round(float(delta_fav[19]), 3),
                             d20_proxy_lower_minus_random_upper=round(
                                 float((proxy_lower - base["upper"])[19]), 3),
                             worst=round(float(delta_fav.min()), 3),
                             worst_proxy_lower_minus_random_upper=round(
                                 float((proxy_lower - base["upper"]).min()), 3),
                             bstar=wb, winp=round(pw, 5), top1=round(float(delta_fav[0]), 3),
                             unsafe_count=unsafe,
                             rho_test=round(float(spearmanr(s, test).statistic), 3),
                             rho_val=round(float(spearmanr(s, val).statistic), 3)))
            print(f"{name:24s}{delta_fav[19]:8.2f}{delta_fav.min():9.2f}{wb:6d}{pw:10.4f}{unsafe:8d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out + ".npz", b_max=B_MAX, **store)
    Path(args.out + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\nwrote {args.out}.npz / .json")


if __name__ == "__main__":
    main()
