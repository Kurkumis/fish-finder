"""Integrity checks and near-duplicate detection for downloaded photos."""
from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 16  # dHash grid: 16 rows x 17 columns -> 256 bits


def inspect_image(path: Path) -> dict:
    """Return basic properties and a difference hash of an image.

    The result has the keys: ok, width, height, file_bytes, dhash.
    `dhash` is an array of 4 uint64 values (256 bits), or None if the file
    is missing or cannot be decoded.
    """
    info = {"ok": False, "width": None, "height": None, "file_bytes": None, "dhash": None}
    if not path.exists():
        return info
    info["file_bytes"] = path.stat().st_size
    try:
        with Image.open(path) as img:
            img.load()  # decode the whole file
            info["width"], info["height"] = img.size
            gray = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    except (OSError, SyntaxError, ValueError):
        return info

    pixels = np.asarray(gray, dtype=np.int16)
    # One bit per pixel: is it brighter than its left neighbour?
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()  # 256 booleans
    info["dhash"] = np.packbits(bits).view(np.uint64)  # 32 bytes -> 4 uint64
    info["ok"] = True
    return info


def find_near_duplicates(
    hashes: np.ndarray, max_distance: int = 10, chunk: int = 256
) -> list[tuple[int, int]]:
    """Return index pairs (i, j), i < j, whose hashes differ in <= max_distance bits.

    `hashes` has shape (n, 4) and dtype uint64. Rows are compared in chunks
    to keep memory usage low.
    """
    pairs = []
    n = len(hashes)
    for start in range(0, n, chunk):
        block = hashes[start : start + chunk]  # (c, 4)
        xor = block[:, None, :] ^ hashes[None, :, :]  # (c, n, 4): differing bits
        distance = np.bitwise_count(xor).sum(axis=2)  # (c, n): Hamming distances
        rows, cols = np.nonzero(distance <= max_distance)
        for r, c in zip(rows, cols):
            i = start + int(r)
            if i < c:  # keep each pair once and skip self-matches
                pairs.append((i, int(c)))
    return pairs


def cluster_pairs(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Group items connected by duplicate pairs.

    Returns an array with a cluster label for each of the n items.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])
