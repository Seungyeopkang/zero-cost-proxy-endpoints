"""NATS-Bench(TSS) API에서 NAS-Bench-201 검증/테스트 정확도를 전수 추출한다.

리뷰 P0-1 대응. NAS-Bench-Suite-Zero 배포본에는 val accuracy만 있으므로
공식 API에서 test accuracy를 직접 읽어 온다. hp='200', is_random=False (시드 평균).

  python3 experiments/fetch_nb201_test.py --root data/nats/NATS-tss-v1_0-3ffb9-simple
산출: data/nb201_acc.npz  (arch_index 순서, 15,625)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exnas.space import all_archs, arch_to_str  # noqa: E402

# (저장 키, NATS dataset 이름, get_more_info 키)
FIELDS = [
    ("c10_valid", "cifar10-valid", "valid-accuracy"),
    ("c10_test", "cifar10", "test-accuracy"),
    ("c100_valid", "cifar100", "valid-accuracy"),
    ("c100_test", "cifar100", "test-accuracy"),
    ("in16_valid", "ImageNet16-120", "valid-accuracy"),
    ("in16_test", "ImageNet16-120", "test-accuracy"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nats/NATS-tss-v1_0-3ffb9-simple")
    ap.add_argument("--out", default="data/nb201_acc.npz")
    args = ap.parse_args()

    from nats_bench import create
    api = create(args.root, "tss", fast_mode=True, verbose=False)

    archs = list(all_archs())
    out = {k: np.full(len(archs), np.nan) for k, _, _ in FIELDS}
    t0 = time.time()
    for i, a in enumerate(archs):
        gi = api.query_index_by_arch(arch_to_str(a))
        cache = {}
        for key, ds, field in FIELDS:
            if ds not in cache:
                cache[ds] = api.get_more_info(gi, ds, hp="200", is_random=False)
            out[key][i] = cache[ds][field]
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(archs)}  {el:.0f}s  eta {el/(i+1)*(len(archs)-i-1):.0f}s",
                  flush=True)
        api.clear_params(gi)

    assert not any(np.isnan(v).any() for v in out.values()), "결측 있음"
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    for k in out:
        print(f"  {k:12s} mean {out[k].mean():.3f}  min {out[k].min():.3f}  max {out[k].max():.3f}")


if __name__ == "__main__":
    main()
