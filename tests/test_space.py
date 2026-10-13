import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exnas import space


def test_space_size():
    assert space.SPACE_SIZE == 15625
    assert len(list(space.all_archs())) == space.SPACE_SIZE


def test_index_roundtrip():
    for idx in [0, 1, 7, 999, 15624]:
        assert space.arch_index(space.index_to_arch(idx)) == idx


def test_index_matches_enumeration_order():
    for i, arch in enumerate(space.all_archs()):
        if i > 200:
            break
        assert space.arch_index(arch) == i


def test_str_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = space.random_arch(rng)
        assert space.str_to_arch(space.arch_to_str(a)) == a


def test_mutate_changes_exactly_one_edge():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = space.random_arch(rng)
        b = space.mutate(a, rng)
        diff = [i for i in range(space.N_EDGES) if a[i] != b[i]]
        assert len(diff) == 1


def test_index_out_of_range():
    with pytest.raises(ValueError):
        space.index_to_arch(space.SPACE_SIZE)
