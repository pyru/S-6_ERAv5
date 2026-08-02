"""Canonical serialization and hashing primitives.

Every identity in TDES (tokenizer, shard, pack, batch, ledger record) is a
sha256 over a *canonical* JSON encoding, so identity is stable across
processes, machines and Python versions.
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
from typing import Any, Iterable, Sequence

ZERO = "0" * 64


def canon(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, unicode preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canon(obj).encode("utf-8"))


def hash_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def hash_tokens(tokens: Sequence[int]) -> str:
    """Hash a token id sequence as fixed-width little-endian uint32."""
    buf = struct.pack("<%dI" % len(tokens), *[int(t) for t in tokens])
    return sha256_hex(buf)


def chain_hash(prev_hash: str, payload: Any) -> str:
    """Hash-chain step: H(prev || H(payload))."""
    return sha256_hex((prev_hash + hash_obj(payload)).encode("utf-8"))


def merkle_root(leaves: Iterable[str]) -> str:
    """Binary merkle root over hex leaves (duplicate-last padding)."""
    level = [l for l in leaves]
    if not level:
        return ZERO
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            sha256_hex((level[i] + level[i + 1]).encode("utf-8"))
            for i in range(0, len(level), 2)
        ]
    return level[0]


def derive_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary parts."""
    return int(hash_obj(list(parts))[:16], 16) & ((1 << 63) - 1)


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(derive_seed(*parts))


def stable_shuffle(items: Sequence[Any], *seed_parts: Any) -> list:
    out = list(items)
    stable_rng(*seed_parts).shuffle(out)
    return out


def ngram_fingerprints(tokens: Sequence[int], n: int = 13, stride: int = 1) -> set:
    """Token n-gram fingerprints used for contamination detection."""
    out = set()
    for i in range(0, max(0, len(tokens) - n + 1), stride):
        out.add(hash_tokens(tokens[i : i + n])[:16])
    return out
