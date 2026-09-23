"""Perceptual hashing and near-duplicate search (numpy + OpenCV, no extra dependencies).

Near-duplicates are found in two stages to keep false positives low on MRI (many slices share a dark background
and a round skull outline):

1. candidate pairs whose 64-bit pHash differ in at most ``max_hamming`` bits (vectorised XOR + popcount);
2. verification by Pearson correlation of 32x32 grayscale thumbnails (``min_corr``).
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _bits_to_uint64(bits: np.ndarray) -> np.uint64:
    return np.uint64(int.from_bytes(np.packbits(bits.astype(np.uint8).ravel()).tobytes(), "big"))


def dhash(gray: np.ndarray, size: int = 8) -> np.uint64:
    """Difference hash: sign of horizontal gradients on a (size x size+1) thumbnail."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return _bits_to_uint64(small[:, 1:] > small[:, :-1])


def phash(gray: np.ndarray, size: int = 8, factor: int = 4) -> np.uint64:
    """DCT perceptual hash: low-frequency DCT coefficients compared with their median."""
    n = size * factor
    small = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = np.asarray(cv2.dct(small), dtype=np.float32)[:size, :size]
    return _bits_to_uint64(low > float(np.median(low)))


def thumbnail_vector(gray: np.ndarray, size: int = 32) -> np.ndarray:
    """Zero-mean, unit-norm thumbnail so that a dot product equals the Pearson correlation."""
    v = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-6 else v


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits per uint64 (numpy>=2 fast path, byte lookup table otherwise)."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    return _POPCOUNT_LUT[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1, dtype=np.uint8)


def hamming_pairs(
    hashes: np.ndarray, max_hamming: int, other: np.ndarray | None = None, chunk: int = 1024
) -> list[tuple[int, int, int]]:
    """All pairs within ``max_hamming`` bits.

    With ``other=None`` returns pairs ``i < j`` inside ``hashes``; otherwise pairs ``(i in hashes, j in other)``.
    """
    a = np.asarray(hashes, dtype=np.uint64)
    b = a if other is None else np.asarray(other, dtype=np.uint64)
    out: list[tuple[int, int, int]] = []
    for start in range(0, a.shape[0], chunk):
        block = a[start : start + chunk]
        dist = popcount64(block[:, None] ^ b[None, :])
        ii, jj = np.nonzero(dist <= max_hamming)
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
            gi = start + i
            if other is None and j <= gi:
                continue
            out.append((gi, j, int(dist[i, j])))
    return out


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self, members: Iterable[int] | None = None) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for i in members if members is not None else range(len(self.parent)):
            buckets.setdefault(self.find(i), []).append(i)
        return [sorted(g) for g in buckets.values()]
