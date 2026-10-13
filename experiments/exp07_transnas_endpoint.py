"""EXP07 — TransNAS-Bench-101에서 끝점 분석 재현 (리뷰 P1-6 대응).

NAS-Bench-201 하나에만 의존한다는 지적에 대응하여, 같은 13종 zero-cost proxy 점수가
공개된 두 개의 **전수 열거 공간**에서 동일한 분석을 반복한다.

  micro : cell 위상 공간 4,096개 (NB-201과 같은 종류)
  macro : 골격(깊이·폭) 공간 3,256개 (zen이 원래 설계 대상으로 삼은 종류)

과제는 정확도가 퍼센트로 정의되는 세 가지만 쓴다 (class_scene, class_object, jigsaw).
autoencoder/normal(SSIM), room_layout(음의 L2), segmentsemantic(mIoU)은 단위가 달라 제외한다.

학습예산 상한은 공간 크기에 비례하여 맞춘다. NB-201은 1000/15625 = 6.4%이므로
여기서는 250을 쓴다 (micro 6.1%, macro 7.7%).

순위 변환은 전 공간에서 정확히 계산한다. NB-201에서 보정 표본 크기의 영향이
0.06%p 이내였으므로(V장 4절) 표본을 쓸 이유가 없다.

  python3 experiments/exp07_transnas_endpoint.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import spearmanr

PROXIES = ("epe_nas", "fisher", "flops", "grad_norm", "grasp", "jacov", "l2_norm",
           "nwot", "params", "plain", "snip", "synflow", "zen")
TASKS = ("class_scene", "class_object", "jigsaw")
SPACES = ("micro", "macro")
B_MAX = 250


def logC(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def random_baseline(acc: np.ndarray, b_max: int) -> np.ndarray:
    n = len(acc)
    a = np.sort(acc)
    i = np.arange(1, n + 1, dtype=np.float64)
    out = np.empty(b_max)
    for b in range(1, b_max + 1):
        lw = np.where(i >= b, logC(i - 1, b - 1) - logC(n, b), -np.inf)
        w = np.exp(lw)
        out[b - 1] = float(np.dot(a, w / w.sum()))
    return out


def winprob(acc: np.ndarray, e: float, b: int) -> float:
    n = len(acc)
    m = int((acc <= e).sum())
    if m < b:
        return 1.0
    return 1.0 - float(np.exp(logC(m, b) - logC(n, b)))


def intervals(mask: np.ndarray) -> str:
    out, st = [], None
    for i, v in enumerate(mask):
        if v and st is None:
            st = i + 1
        if not v and st is not None:
            out.append((st, i)); st = None
    if st is not None:
        out.append((st, len(mask)))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out) if out else "-"


def endpoint(acc, s, tie="opt"):
    key = -acc if tie == "opt" else acc
    return np.maximum.accumulate(acc[np.lexsort((key, -s))])[:B_MAX]


def combos(raw: np.ndarray) -> dict[str, np.ndarray]:
    """전 공간 기준 결합 규칙. 표현(점수/순위) x 결합함수."""
    z = (raw - raw.mean(0)) / np.where(raw.std(0) > 0, raw.std(0), 1.0)
    n = len(raw)
    pr = np.apply_along_axis(lambda c: (np.argsort(np.argsort(c)) + 1) / n, 0, raw)
    return {
        "mean of standardized scores": z.mean(1),
        "mean percentile rank": pr.mean(1),
        "AZ-NAS-style non-linear": np.log(np.clip(pr, 1e-12, None)).sum(1),
        "worst rank": pr.min(1),
        "soft minimum of z": -logsumexp(-z, axis=1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp07_transnas")
    args = ap.parse_args()

    rows, store = [], {}
    for space in SPACES:
        data = json.load(open(f"data/zc_transbench101_{space}.json"))
        for task in TASKS:
            recs = data[task]
            keys = list(recs)
            acc = np.array([recs[k]["val_accuracy"] for k in keys], dtype=np.float64)
            raw = np.array([[recs[k][p]["score"] for p in PROXIES] for k in keys],
                           dtype=np.float64)
            bad = ~np.isfinite(raw)
            if bad.any():
                # NBSZ 일부 셀에 inf/nan 이 있다. 해당 proxy의 최솟값으로 대체하고 보고한다.
                for j in range(raw.shape[1]):
                    m = bad[:, j]
                    if m.any():
                        raw[m, j] = np.nanmin(raw[~m, j])
            base = random_baseline(acc, B_MAX)
            store[f"{space}/{task}/random"] = base
            n = len(acc)
            print(f"\n=== {space} / {task}  n={n}  acc {acc.min():.2f}..{acc.max():.2f} "
                  f"mean {acc.mean():.2f}  nonfinite proxy cells {int(bad.sum())}")
            scorers = {p: raw[:, j] for j, p in enumerate(PROXIES)}
            scorers.update(combos(raw))
            print(f"{'scorer':30s}{'d20':>8}{'worst':>9}{'@B':>6}{'P@worst':>9}  unsafe(<-1pp)")
            for name, s in scorers.items():
                eo, ep = endpoint(acc, s, "opt"), endpoint(acc, s, "pes")
                do, dp = eo - base, ep - base
                wb = int(do.argmin()) + 1
                pw = winprob(acc, float(eo[wb - 1]), wb)
                store[f"{space}/{task}/{name}"] = do
                rows.append(dict(space=space, task=task, scorer=name,
                                 d20=round(float(do[19]), 3),
                                 worst=round(float(do.min()), 3), bstar=wb,
                                 worst_pes=round(float(dp.min()), 3),
                                 top1=round(float(do[0]), 3), winp=round(pw, 5),
                                 unsafe=intervals(do < -1.0),
                                 rho=round(float(spearmanr(s, acc).statistic), 3)))
                print(f"{name:30s}{do[19]:8.2f}{do.min():9.2f}{wb:6d}{pw:9.4f}  "
                      f"{intervals(do < -1.0)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out + ".npz", b_max=B_MAX, **store)
    Path(args.out + ".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nwrote {args.out}.npz / .json")


if __name__ == "__main__":
    main()
