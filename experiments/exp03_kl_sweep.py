"""EXP03 — 최적화 강도(KL) 스윕. C1의 kill-test. NB-201 실데이터.

고정 학습예산 B에서 최적화 강도를 두 손잡이로 올린다:
  - 선택압: EA의 sample_size (tournament 크기)
  - 후보 수: N
두 손잡이를 KL(개체군 || 균등) 하나의 축으로 통합해 최종 acc와의 관계를 본다.

사전 등록 kill 기준 (RESEARCH_PLAN.md §5):
  KILL-C1  : 최고 성능 KL 지점과 최대 KL 지점의 차이가 모든 조합에서 2 SEM 이내 -> C1 폐기
  KILL-C1b : 내부 최적점이 있어도 정점->최대KL 하락폭이 3개 데이터셋 모두 0.2pp 미만 -> 폐기
  GATE-C1  : anti-oracle이 KL에 따라 단조 악화하지 않으면 파이프라인 결함

실행:
    python3 experiments/exp03_kl_sweep.py --datasets cifar10 --seeds 20
    python3 experiments/exp03_kl_sweep.py --datasets cifar10 cifar100 ImageNet16-120 --seeds 20 --only-random-proxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas.aggregate import (  # noqa: E402
    AZNASLinear, AZNASNonLinear, AntiOracle, LinearWeighted, RandomProxy,
    SingleProxy, SoftConsensus, WorstRank,
)
from exnas.benchmark import NB201Benchmark  # noqa: E402
from exnas.logging_utils import append_jsonl, config_hash, run_header  # noqa: E402
from exnas.pipeline import make_calibrator, run_pipeline  # noqa: E402

N_GRID = [200, 500, 1000, 2000, 5000]
PRESSURE_GRID = [2, 5, 10, 25, 50]  # EA sample_size. 클수록 선택압 강함


def build_aggregators(bench, n_calib: int, focus: bool):
    idx = {p: i for i, p in enumerate(bench.proxy_names)}
    aggs = [
        SingleProxy(idx["synflow"]),
        SingleProxy(idx["nwot"]),
        AZNASNonLinear(n_calib=n_calib),
        AZNASLinear(n_calib=n_calib),
        AntiOracle(SingleProxy(idx["synflow"])),   # GATE-C1
        RandomProxy(),                              # 하한선
    ]
    if not focus:
        # 붕괴하는 proxy를 반드시 포함해야 한다. focus 모드는 synflow/nwot만 봐서
        # '안 무너지는 것만' 시험하는 오류가 있었다 (F9 참조).
        others = [p for p in bench.proxy_names if p not in ("synflow", "nwot")]
        aggs += [SingleProxy(idx[p]) for p in others]
        aggs += [LinearWeighted(), SoftConsensus(p=4.0), WorstRank()]
    return aggs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10"],
                    choices=list(NB201Benchmark.DATASETS))
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--population-size", type=int, default=50)
    ap.add_argument("--focus", action="store_true", help="결합기 6종만 (빠른 1차 확인)")
    ap.add_argument("--only-random-proxy", action="store_true",
                    help="random-score 대조군만 실행 (기존 전체 실험의 해당 행 재현용)")
    ap.add_argument("--data", default="data/zc_nasbench201.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    config = vars(args) | {"experiment": "exp03_kl_sweep",
                           "n_grid": N_GRID, "pressure_grid": PRESSURE_GRID}
    h = config_hash(config)
    out = Path(args.out) / f"exp03_{h}.jsonl"
    append_jsonl(out, [run_header(config)])

    records = []
    for ds in args.datasets:
        bench = NB201Benchmark(ds, args.data)
        aggs = build_aggregators(bench, args.n_calib, args.focus)
        if args.only_random_proxy:
            aggs = [agg for agg in aggs if agg.name == "random_proxy"]
        for seed in range(args.seeds):
            cal = make_calibrator(bench, np.random.default_rng(10_000 + seed), args.n_calib)
            for agg in aggs:
                for n in N_GRID:
                    # KL ~ 0 기준점
                    r = run_pipeline(bench, agg, cal, "random", n, args.budget, seed)
                    records.append(r.to_dict() | {"_type": "run", "config_hash": h,
                                                  "dataset": ds, "pressure": 0})
                    for s in PRESSURE_GRID:
                        r = run_pipeline(bench, agg, cal, "ea", n, args.budget, seed,
                                         population_size=args.population_size, sample_size=s)
                        records.append(r.to_dict() | {"_type": "run", "config_hash": h,
                                                      "dataset": ds, "pressure": s})
            print(f"{ds} seed {seed}: {len(records)} runs", flush=True)

    append_jsonl(out, records)
    print(f"\nwrote {len(records)} records -> {out}")
    return out


if __name__ == "__main__":
    main()
