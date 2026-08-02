"""Frozen byte-level BPE tokenizer (Session 2 contract).

Guarantees
----------
* Lossless: ``decode(encode(t)) == normalize(t)`` for any input, because the
  base alphabet is the 256 byte values.
* Deterministic: merge learning uses count-desc then lexicographic tie-breaks,
  so the same corpus always yields the same vocabulary.
* Indic-safe: NFC only.  We never apply NFKC (it destroys Indic presentation
  forms), never casefold, and never strip ZWJ (U+200D) / ZWNJ (U+200C), which
  are semantically load-bearing in Devanagari and other Indic scripts.
* Frozen: the tokenizer is serialized once and identified by ``tokenizer_hash``.
  Every shard records that hash; the loader refuses to mix hashes.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from .config import PATHS, VOCAB_SIZE
from .hashing import hash_obj

# id space: 0..255 raw bytes, then canonical specials, then learned merges
SPECIALS: List[str] = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|user|>", "<|assistant|>", "<|tool_call|>", "<|tool_obs|>",
    "<|think|>", "<|answer|>", "<|sep|>",
]
SPECIAL_BASE = 256
SPECIAL_IDS: Dict[str, int] = {s: SPECIAL_BASE + i for i, s in enumerate(SPECIALS)}
PAD = SPECIAL_IDS["<|pad|>"]
BOS = SPECIAL_IDS["<|bos|>"]
EOS = SPECIAL_IDS["<|eos|>"]
MERGE_BASE = SPECIAL_BASE + len(SPECIALS)

ROLE_TOKEN = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool_call": "<|tool_call|>",
    "tool_obs": "<|tool_obs|>",
    "think": "<|think|>",
    "answer": "<|answer|>",
    "text": "",
}

_PRETOK = re.compile(r"\s+|\w+|[^\s\w]", re.UNICODE)

# codepoints that must survive normalization untouched
ZERO_WIDTH = {"‌", "‍"}


def normalize(text: str) -> str:
    """Indic-safe normalization: NFC, CRLF folding, no NFKC/casefold/ZW stripping."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out = unicodedata.normalize("NFC", text)
    return out


class Tokenizer:
    def __init__(self, merges: List[Tuple[bytes, bytes]], vocab_size: int):
        self.merges = merges
        self.vocab_size = vocab_size
        self.rank: Dict[Tuple[bytes, bytes], int] = {p: i for i, p in enumerate(merges)}
        self.merge_id: Dict[Tuple[bytes, bytes], int] = {
            p: MERGE_BASE + i for i, p in enumerate(merges)
        }
        self.id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for p, i in self.merge_id.items():
            self.id_to_bytes[i] = p[0] + p[1]
        self.bytes_to_id: Dict[bytes, int] = {v: k for k, v in self.id_to_bytes.items()}
        for s, i in SPECIAL_IDS.items():
            self.id_to_bytes[i] = s.encode("utf-8")
        self.special_ids = set(SPECIAL_IDS.values())

    # ---------------------------------------------------------------- spec --
    def spec(self) -> dict:
        return {
            "format": "tdes-byte-bpe",
            "version": "1.0.0",
            "normalization": {
                "form": "NFC",
                "nfkc": False,
                "casefold": False,
                "strip_zero_width": False,
                "crlf_folded": True,
            },
            "specials": SPECIALS,
            "special_base": SPECIAL_BASE,
            "merge_base": MERGE_BASE,
            "vocab_size": self.vocab_size,
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
        }

    @property
    def tokenizer_hash(self) -> str:
        return hash_obj(self.spec())

    # -------------------------------------------------------------- encode --
    def _encode_chunk(self, chunk: bytes) -> List[int]:
        parts: List[bytes] = [bytes([b]) for b in chunk]
        while len(parts) > 1:
            best, best_rank = -1, None
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best_rank is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return [self.bytes_to_id[p] for p in parts]

    def encode(self, text: str, normalize_input: bool = True) -> List[int]:
        if normalize_input:
            text = normalize(text)
        out: List[int] = []
        for m in _PRETOK.finditer(text):
            out.extend(self._encode_chunk(m.group(0).encode("utf-8")))
        return out

    def encode_special(self, name: str) -> int:
        return SPECIAL_IDS[name]

    # -------------------------------------------------------------- decode --
    def decode(self, ids: Sequence[int], skip_specials: bool = False) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in self.special_ids:
                if skip_specials:
                    continue
                buf.extend(self.id_to_bytes[i])
            else:
                buf.extend(self.id_to_bytes.get(i, b"\xef\xbf\xbd"))
        return buf.decode("utf-8", errors="replace")

    def preview(self, tid: int) -> str:
        b = self.id_to_bytes.get(int(tid), b"?")
        return b.decode("utf-8", errors="replace")

    # ----------------------------------------------------------- persistence --
    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(PATHS["manifests"], "tokenizer.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        doc = {"tokenizer_hash": self.tokenizer_hash, "spec": self.spec()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        return path

    @staticmethod
    def load(path: str | None = None) -> "Tokenizer":
        path = path or os.path.join(PATHS["manifests"], "tokenizer.json")
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        spec = doc["spec"]
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in spec["merges"]]
        tok = Tokenizer(merges, spec["vocab_size"])
        if tok.tokenizer_hash != doc["tokenizer_hash"]:
            raise ValueError("tokenizer hash mismatch on load: file was mutated")
        return tok


def train_tokenizer(texts: Sequence[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Deterministic byte-level BPE training."""
    freq: Dict[Tuple[bytes, ...], int] = {}
    for t in texts:
        for m in _PRETOK.finditer(normalize(t)):
            key = tuple(bytes([b]) for b in m.group(0).encode("utf-8"))
            freq[key] = freq.get(key, 0) + 1

    n_merges = max(0, vocab_size - MERGE_BASE)
    merges: List[Tuple[bytes, bytes]] = []
    words = {k: list(k) for k in freq}

    for _ in range(n_merges):
        pair_counts: Dict[Tuple[bytes, bytes], int] = {}
        for key, parts in words.items():
            c = freq[key]
            for i in range(len(parts) - 1):
                p = (parts[i], parts[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + c
        if not pair_counts:
            break
        # deterministic: highest count, then lexicographic on the byte pair
        best = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[0]
        if pair_counts[best] < 2:
            break
        merges.append(best)
        a, b = best
        for key, parts in words.items():
            i = 0
            new: List[bytes] = []
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(parts[i])
                    i += 1
            words[key] = new

    return Tokenizer(merges, MERGE_BASE + len(merges))


def tokenize_document(tok: Tokenizer, doc: dict) -> dict:
    """Tokenize a document into role-tagged segments plus a flat token stream.

    Returns ``{'tokens': [...], 'roles': [...], 'loss': [...]}`` where ``roles``
    and ``loss`` are per-token and carry the Session 1 loss contract:
    context roles (user prompt, tool observation) are not loss-bearing.
    """
    tokens: List[int] = []
    roles: List[str] = []
    loss: List[int] = []
    for seg in doc["segments"]:
        role = seg["role"]
        marker = ROLE_TOKEN.get(role, "")
        if marker:
            tokens.append(SPECIAL_IDS[marker])
            roles.append(role)
            loss.append(1 if seg["loss"] else 0)
        ids = tok.encode(seg["text"])
        tokens.extend(ids)
        roles.extend([role] * len(ids))
        loss.extend([1 if seg["loss"] else 0] * len(ids))
    tokens.append(EOS)
    roles.append("eos")
    loss.append(1)
    return {"tokens": tokens, "roles": roles, "loss": loss}
