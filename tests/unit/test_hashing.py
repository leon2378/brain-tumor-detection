from __future__ import annotations

import cv2
import numpy as np

from btd.data.hashing import UnionFind, dhash, hamming_pairs, phash, popcount64, thumbnail_vector
from btd.data.synthetic import make_slice


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def test_hashes_are_robust_to_recompression_and_discriminative() -> None:
    rng = np.random.default_rng(0)
    a, _ = make_slice(rng, "glioma", 256)
    b, _ = make_slice(rng, "pituitary", 256)
    a2 = _jpeg_roundtrip(a, 40)
    d_same = int(popcount64(np.array([phash(a) ^ phash(a2)], np.uint64))[0])
    d_diff = int(popcount64(np.array([phash(a) ^ phash(b)], np.uint64))[0])
    assert d_same <= 4 < d_diff
    assert float(thumbnail_vector(a) @ thumbnail_vector(a2)) > 0.99
    assert int(popcount64(np.array([dhash(a) ^ dhash(a2)], np.uint64))[0]) <= 6


def test_popcount_fallback_matches_numpy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    x = np.array([0, 1, 2**63, 2**64 - 1, 0xF0F0], dtype=np.uint64)
    fast = popcount64(x).tolist()
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    assert popcount64(x).tolist() == fast == [0, 1, 1, 64, 8]


def test_hamming_pairs_self_and_cross() -> None:
    h = np.array([0b0000, 0b0001, 0b1111, 0b0000], dtype=np.uint64)
    pairs = hamming_pairs(h, max_hamming=1, chunk=2)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (0, 3), (1, 3)]
    cross = hamming_pairs(h[:2], max_hamming=0, other=h[2:])
    assert [(i, j) for i, j, _ in cross] == [(0, 1)]


def test_union_find() -> None:
    uf = UnionFind(5)
    uf.union(0, 3)
    uf.union(3, 4)
    groups = sorted(uf.groups())
    assert [0, 3, 4] in groups and [1] in groups and [2] in groups
