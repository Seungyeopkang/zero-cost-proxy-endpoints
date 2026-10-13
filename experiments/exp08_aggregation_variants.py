"""Reproduce Table 5(b): representation and aggregation controls on NB-201.

All 13 published proxy signs are kept. The calibration set is the same
seed-zero, 2,000-architecture sample without replacement as exp05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas.aggregate import Calibrator, _ascending_percentile  # noqa: E402
from exnas.benchmark import NB201Benchmark  # noqa: E402
from exnas.space import all_archs  # noqa: E402
from exp05_endpoint_curve import random_baseline_curve  # noqa: E402


def scores_by_variant(raw: np.ndarray, sample: np.ndarray) -> dict[str, np.ndarray]:
    """The eight rows of Table 5(b), in their published order."""
    calibration = raw[sample]
    cal = Calibrator(calibration)
    z, upper_fraction = cal.transform(raw)
    percentile = _ascending_percentile(upper_fraction, len(sample))

    median = np.median(calibration, axis=0)
    q25, q75 = np.percentile(calibration, [25, 75], axis=0)
    iqr = q75 - q25
    robust = (raw - median) / np.where(iqr > 0, iqr, 1.0)

    return {
        "arithmetic mean / raw score": raw.mean(axis=1),
        "arithmetic mean / standardized score": z.mean(axis=1),
        "arithmetic mean / robust standardized score": robust.mean(axis=1),
        "arithmetic mean / standardized score clipped at ±3": np.clip(z, -3, 3).mean(axis=1),
        "arithmetic mean / percentile rank": percentile.mean(axis=1),
        "soft minimum / standardized score": -logsumexp(-z, axis=1),
        "logarithmic sum / percentile rank": np.log(percentile).sum(axis=1),
        "minimum / percentile rank": percentile.min(axis=1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/zc_nasbench201.json")
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--b-max", type=int, default=1000)
    ap.add_argument("--out", default="results/exp08_aggregation_variants.json")
    args = ap.parse_args()

    archs = list(all_archs())
    sample = np.random.default_rng(0).choice(len(archs), args.n_calib, replace=False)
    rows = []
    for ds in NB201Benchmark.DATASETS:
        bench = NB201Benchmark(ds, args.data)
        raw = bench.proxy_scores(archs)
        acc = bench.all_acc()
        base = random_baseline_curve(acc, args.b_max)
        for name, score in scores_by_variant(raw, sample).items():
            order = np.argsort(-score)
            delta = np.maximum.accumulate(acc[order])[: args.b_max] - base
            row = {"dataset": ds, "variant": name, "worst_delta": float(delta.min()),
                   "worst_b": int(delta.argmin()) + 1}
            rows.append(row)
            print(f"{ds:16s} {name:53s} {row['worst_delta']:+.6f} at {row['worst_b']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"settings": {"n_calib": args.n_calib, "calibration_seed": 0,
                                            "replace": False, "ddof": 0, "b_max": args.b_max},
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
