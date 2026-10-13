"""NAS-Bench-201 탐색 공간 인코딩.

아키텍처는 6개 edge, 각 edge가 5개 op 중 하나 -> 5**6 = 15,625개.
edge 순서는 NAS-Bench-201 문자열 표현과 동일하다:
    0: node1 <- node0
    1: node2 <- node0
    2: node2 <- node1
    3: node3 <- node0
    4: node3 <- node1
    5: node3 <- node2
"""

from __future__ import annotations

import itertools
from typing import Iterator, Sequence

import numpy as np

# 내부 인코딩은 NAS-Bench-Suite-Zero(NASLib) 규약을 따른다. 데이터 파일의 키가 그 규약이며,
# 공식 NAS-Bench-201 문자열 표현과는 op 라벨 순서와 edge 순서가 모두 다르다.
# (2026-08-31 검증: 고유 val accuracy 703건에서 아래 대응이 100% 일치)
OPS: tuple[str, ...] = (
    "skip_connect",     # 0
    "none",             # 1
    "nor_conv_3x3",     # 2
    "nor_conv_1x1",     # 3
    "avg_pool_3x3",     # 4
)

#: 공식 NAS-Bench-201 op 순서 (문자열 표현이 쓰는 순서)
OFFICIAL_OPS: tuple[str, ...] = (
    "none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3",
)

#: 내부(NBSZ) edge 순서를 공식 문자열 edge 순서로 보내는 위치 대응.
#: 내부 0..5 = (1<-0), (2<-0), (3<-0), (2<-1), (3<-1), (3<-2)
#: 공식 0..5 = (1<-0), (2<-0), (2<-1), (3<-0), (3<-1), (3<-2)
EDGE_TO_OFFICIAL: tuple[int, ...] = (0, 1, 3, 2, 4, 5)

#: op 이름으로 본 편의 상수
OP_NONE = OPS.index("none")
OP_SKIP = OPS.index("skip_connect")
CONV_OPS = (OPS.index("nor_conv_1x1"), OPS.index("nor_conv_3x3"))

N_EDGES = 6
N_OPS = len(OPS)
SPACE_SIZE = N_OPS**N_EDGES  # 15625

Arch = tuple[int, ...]


def all_archs() -> Iterator[Arch]:
    """탐색 공간 전체를 사전순으로 열거한다."""
    return itertools.product(range(N_OPS), repeat=N_EDGES)


def arch_index(arch: Arch) -> int:
    """아키텍처를 [0, SPACE_SIZE) 정수로 인코딩. all_archs() 순서와 일치."""
    idx = 0
    for op in arch:
        idx = idx * N_OPS + op
    return idx


def index_to_arch(idx: int) -> Arch:
    """arch_index의 역함수."""
    if not 0 <= idx < SPACE_SIZE:
        raise ValueError(f"index out of range: {idx}")
    ops = []
    for _ in range(N_EDGES):
        idx, op = divmod(idx, N_OPS)
        ops.append(op)
    return tuple(reversed(ops))


def random_arch(rng: np.random.Generator) -> Arch:
    return tuple(int(x) for x in rng.integers(0, N_OPS, size=N_EDGES))


def mutate(arch: Arch, rng: np.random.Generator) -> Arch:
    """edge 하나를 골라 현재와 다른 op로 바꾼다 (aging evolution의 표준 mutation)."""
    edge = int(rng.integers(0, N_EDGES))
    choices = [op for op in range(N_OPS) if op != arch[edge]]
    new_op = int(rng.choice(choices))
    out = list(arch)
    out[edge] = new_op
    return tuple(out)


def arch_to_str(arch: Sequence[int]) -> str:
    """NAS-Bench-201 / NATS-Bench API가 받는 공식 문자열 표현으로 변환.

    내부 인코딩(NBSZ)과 공식 표현은 op 라벨 순서와 edge 순서가 모두 다르므로
    두 가지를 함께 변환한다. 예: '|nor_conv_3x3~0|+|none~0|skip_connect~1|+|...|'
    """
    o = [""] * N_EDGES
    for e, op in enumerate(arch):
        o[EDGE_TO_OFFICIAL[e]] = OPS[op]
    return (
        f"|{o[0]}~0|+"
        f"|{o[1]}~0|{o[2]}~1|+"
        f"|{o[3]}~0|{o[4]}~1|{o[5]}~2|"
    )


def str_to_arch(s: str) -> Arch:
    """arch_to_str의 역함수. 공식 edge 순서를 내부(NBSZ) 순서로 되돌린다."""
    official: list[int] = []
    for node_block in s.split("+"):
        for token in node_block.strip("|").split("|"):
            if not token:
                continue
            name = token.split("~")[0]
            official.append(OPS.index(name))
    if len(official) != N_EDGES:
        raise ValueError(f"expected {N_EDGES} ops, parsed {len(official)} from {s!r}")
    return tuple(official[EDGE_TO_OFFICIAL[e]] for e in range(N_EDGES))


def edge_op_matrix(archs: Sequence[Arch]) -> np.ndarray:
    """(n_arch, N_EDGES) 정수 행렬. 개체군 분포 통계용."""
    return np.asarray(archs, dtype=np.int64).reshape(len(archs), N_EDGES)
