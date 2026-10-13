"""C1 재분석 — 설정 단위 대응 분석 (그림 7 / 표 7의 SEM 재설계).

기존 analyze_c1.py 문제:
  - KL 분위수 bin 안에 여러 (N, tournament) 설정을 섞고, bin 내 관측치를 독립으로 보고 SEM 계산.
  - 20개 시드가 설정을 공유하므로 의사반복(pseudo-replication)이며 SEM이 과소추정된다.
  - 정점(peak)을 같은 데이터에서 고르고 그 정점의 우위를 그대로 보고하므로 승자의 저주가 낀다.

여기서의 설계:
  - 분석 단위 = 설정 (dataset, aggregator, mode, N, pressure). 각 설정에 시드 0..19.
  - 2겹 교차적합: 시드 0-9로 정점을 고르고 10-19에서 효과를 재고, 반대로도 한 번 더 한다.
    각 시드는 자신이 정점 선택에 쓰이지 않은 쪽에서만 평가되므로 승자의 저주가 제거된다.
  - 최대 KL 설정은 선택 없이 결정되므로 그대로 쓴다.
  - 대응 차이 Δ_s = acc[peak(-s), s] − acc[maxKL, s] 20개에 대하여
    대응 t 검정(주)과 Wilcoxon signed-rank(보조)을 쓰고 Holm 보정을 60개 조합 전체에 적용한다.
  - 통제 대비: N 고정 후 선택압 2 vs 50, 선택압 고정 후 N 200 vs 5000 을 각각 대응 비교.

  python3 experiments/analyze_c1_paired.py results/exp03_e1c77996195e.jsonl results/exp03_f1112324f45d.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SEL_SEEDS = list(range(10))
EVAL_SEEDS = list(range(10, 20))
DS = ("cifar10", "cifar100", "ImageNet16-120")


def load(paths):
    """Read run rows in file order; later files can supersede corrected run keys."""
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("_type") == "run":
                    rows.append(r)
    return rows


def index(rows):
    """(dataset, aggregator) -> {config: {'acc': np.array(20), 'kl': float}}"""
    acc = defaultdict(lambda: defaultdict(dict))
    kl = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["dataset"], r["aggregator"])
        cfg = (r["mode"], r["n_candidates"], r["pressure"])
        acc[key][cfg][r["seed"]] = r["best_acc"]
        kl[key][cfg][r["seed"]] = r["population_kl"]
    out = {}
    for key, cfgs in acc.items():
        d = {}
        for cfg, per_seed in cfgs.items():
            if len(per_seed) != 20:
                continue
            a = np.array([per_seed[s] for s in range(20)])
            k = float(np.mean([kl[key][cfg][s] for s in range(20)]))
            d[cfg] = {"acc": a, "kl": k}
        out[key] = d
    return out


def paired(delta):
    """대응 표본 요약. (평균, 95% CI 반폭, 대응 t p, Wilcoxon p)

    Wilcoxon 은 n=20 에서 최소 p 가 2/2^20 이라 Holm 보정을 통과할 수 있으나
    분포가 완전히 한쪽으로 쏠린 경우에도 이산성 하한에 걸린다. 주 검정은 대응 t 로 두고
    Wilcoxon 은 정규성 가정에 대한 보조 지표로만 보고한다.
    """
    d = np.asarray(delta, dtype=float)
    n = len(d)
    m = float(d.mean())
    if np.allclose(d, d[0]):
        return m, 0.0, (0.0 if m != 0 else 1.0), 1.0
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
    pt = float(stats.ttest_1samp(d, 0.0).pvalue)
    pw = float(stats.wilcoxon(d).pvalue)
    return m, half, pt, pw


def holm(pvals):
    """Holm-Bonferroni 보정된 p."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def crossfit_delta(cfgs, max_kl_cfg):
    """2겹 교차적합 대응 차이 20개. 각 시드는 자신을 뺀 반쪽에서 고른 정점으로 평가된다."""
    cand = [c for c in cfgs if c != max_kl_cfg]
    out = np.empty(20)
    picks = []
    for sel, ev in ((SEL_SEEDS, EVAL_SEEDS), (EVAL_SEEDS, SEL_SEEDS)):
        peak = max(cand, key=lambda c: cfgs[c]["acc"][sel].mean())
        picks.append(peak)
        out[ev] = cfgs[peak]["acc"][ev] - cfgs[max_kl_cfg]["acc"][ev]
    return out, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/c1_paired.json",
                    help="분석 JSON 출력 경로")
    args = ap.parse_args()
    data = index(load([Path(f) for f in args.files]))

    rows = []
    for (ds, agg), cfgs in sorted(data.items()):
        if len(cfgs) < 2:
            continue
        max_kl_cfg = max(cfgs, key=lambda c: cfgs[c]["kl"])
        d_out, picks = crossfit_delta(cfgs, max_kl_cfg)
        m, half, pt, pw = paired(d_out)
        # 비교용: 같은 데이터에서 정점을 고른 순진한 추정 (승자의 저주 포함)
        peak_in = max(cfgs, key=lambda c: cfgs[c]["acc"].mean())
        naive = float(cfgs[peak_in]["acc"].mean() - cfgs[max_kl_cfg]["acc"].mean())
        rows.append(dict(dataset=ds, aggregator=agg, peaks=picks, max_kl_cfg=max_kl_cfg,
                         kl_max=cfgs[max_kl_cfg]["kl"],
                         drop=m, ci=half, p_t=pt, p_wilcoxon=pw, naive=naive))

    adj = holm(np.array([r["p_t"] for r in rows]))
    adj_w = holm(np.array([r["p_wilcoxon"] for r in rows]))
    for r, a, aw in zip(rows, adj, adj_w):
        r["p_holm"] = float(a)
        r["p_holm_wilcoxon"] = float(aw)

    print(f"{len(rows)} (dataset, scorer) combinations, split-seed paired contrast\n")
    hdr = f"{'dataset':16s}{'scorer':22s}{'drop':>8}{'95% CI':>16}{'p_holm':>10}{'naive':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sig = "*" if r["p_holm"] < 0.05 else " "
        print(f"{r['dataset']:16s}{r['aggregator']:22s}{r['drop']:8.2f}"
              f"{'[%.2f, %.2f]' % (r['drop']-r['ci'], r['drop']+r['ci']):>16}"
              f"{r['p_holm']:10.4f}{sig}{r['naive']:7.2f}")

    n_sig = sum(r["p_holm"] < 0.05 for r in rows)
    n_sig_w = sum(r["p_holm_wilcoxon"] < 0.05 for r in rows)
    n_pos = sum(r["p_holm"] < 0.05 and r["drop"] > 0 for r in rows)
    print(f"\nHolm 보정 후 유의한 조합: {n_sig} / {len(rows)} (대응 t), "
          f"{n_sig_w} / {len(rows)} (Wilcoxon), 그중 하락 방향 {n_pos}")
    for ds in DS:
        sub = [r for r in rows if r["dataset"] == ds]
        best = max(sub, key=lambda r: r["drop"])
        print(f"  {ds:16s} 최대 하락 {best['drop']:.2f} pp "
              f"[{best['drop']-best['ci']:.2f}, {best['drop']+best['ci']:.2f}] "
              f"({best['aggregator']}), 순진 추정 {best['naive']:.2f}")

    # ---- 통제 대비: 축을 하나씩 고정한다
    print("\n축을 하나씩 고정한 대응 대비 (전체 20시드, 층별 중앙값)")
    for label, fixed, vary in (("N 고정, 선택압 2 -> 50", "n", "p"),
                               ("선택압 고정, N 200 -> 5000", "p", "n")):
        print(f"\n  [{label}]")
        for ds in DS:
            per_scorer = {}
            for (d, agg), cfgs in data.items():
                if d != ds:
                    continue
                deltas = []
                if fixed == "n":
                    for n in (200, 500, 1000, 2000, 5000):
                        a, b = ("ea", n, 2), ("ea", n, 50)
                        if a in cfgs and b in cfgs:
                            deltas.append(float((cfgs[a]["acc"] - cfgs[b]["acc"]).mean()))
                else:
                    for pr in (2, 5, 10, 25, 50):
                        a, b = ("ea", 200, pr), ("ea", 5000, pr)
                        if a in cfgs and b in cfgs:
                            deltas.append(float((cfgs[a]["acc"] - cfgs[b]["acc"]).mean()))
                if deltas:
                    per_scorer[agg] = float(np.median(deltas))
            worst = max(per_scorer.items(), key=lambda kv: kv[1])
            flat = {k: v for k, v in per_scorer.items()
                    if k in ("single[11]", "aznas_nl", "aznas_l", "mean_rank", "worst_rank",
                             "soft_consensus[p=4]")}
            print(f"    {ds:16s} 최대 {worst[1]:+.2f} pp ({worst[0]}), "
                  f"synflow {per_scorer.get('single[11]', float('nan')):+.2f}, "
                  f"aznas_nl {per_scorer.get('aznas_nl', float('nan')):+.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
