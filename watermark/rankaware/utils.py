# -*- coding: utf-8 -*-


import hashlib
import torch

# -------------------- Hashing (64-bit via Python ints) --------------------

def hash64(x: int) -> int:
    """Return a 64-bit nonnegative hash of integer x."""
    return int.from_bytes(
        hashlib.sha256(int(x).to_bytes(8, "little", signed=False)).digest()[:8],
        "little",
        signed=False,
    )

def pair_hash64(a: int, b: int) -> int:
    """Return a 64-bit nonnegative hash of (a,b) packed into 64 bits."""
    x = ((int(a) & ((1 << 32) - 1)) << 32) ^ (int(b) & ((1 << 32) - 1))
    return hash64(x)

# -------------------- Permutation / Green mask ----------------------------

# Fallback multiplier for an LCG-like mixing (fits in signed 64)
_A64 = 6364136223846793005 % (1 << 62)
_MASK62 = (1 << 62) - 1  # keep numbers positive in signed 64-bit range

def green_mask_from_seed(seed: int, vocab_size: int, m: int, device: str,
                         method: str = "mix-sort") -> torch.Tensor:
    """
    Build a boolean mask of length vocab_size with exactly m greens.
    method:
      - "mix-sort": compute key = (a*i + seed) & MASK62; argsort; take first m
      - "randperm": CPU torch.randperm with a seeded Generator (most compatible)
    """
    if method == "randperm":
        # Deterministic CPU permutation, avoids dtype quirks.
        g = torch.Generator(device="cpu")
        # torch Generator seeds are 32-bit; fold the 64-bit seed down safely
        g.manual_seed(int(seed & 0x7FFFFFFF))
        perm_cpu = torch.randperm(vocab_size, generator=g)  # on CPU
        idx = perm_cpu[:m].to(device)
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        mask[idx] = True
        return mask

    # Default: mix-sort on the target device using int64
    idx = torch.arange(vocab_size, dtype=torch.int64, device=device)
    # key = (a * i + seed) & MASK62, all in signed 64-bit domain
    key = (idx * int(_A64) + int(seed & _MASK62)) & _MASK62  # int64 ops
    perm = torch.argsort(key)  # ascending
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    mask[perm[:m]] = True
    return mask


