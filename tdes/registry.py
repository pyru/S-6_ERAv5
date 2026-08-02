"""Evaluation / validation registry and the training firewall (Session 3+4).

Test data is *registered so that it can be excluded*.  The registry holds
content hashes, benchmark ids, version tags, canary strings and contamination
fingerprints, and every read of held-out data is written to an access log.

Two enforcement points:
  1. Admission time - a candidate document that overlaps registered evaluation
     data (or is registered evaluation data) can never become a shard.
  2. Serving time  - every packed, loss-bearing batch is scanned against the
     token-level fingerprint set before it is handed to the optimizer.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .config import PATHS
from .corpus import CANARY
from .hashing import hash_obj, hash_text, hash_tokens
from .tokenizer import Tokenizer, normalize, tokenize_document

CHAR_WINDOW = 48
CHAR_STRIDE = 16
TOKEN_NGRAM = 13


class EvalRegistry:
    def __init__(self):
        self.entries: Dict[str, dict] = {}
        self.char_fp: Dict[str, str] = {}      # fingerprint -> doc_id
        self.token_fp: Dict[str, str] = {}
        self.text_hashes: Dict[str, str] = {}
        self.canaries: List[str] = [CANARY]
        self.access_log: List[dict] = []
        self.admission_log: List[dict] = []
        self.firewall_events: List[dict] = []

    # ------------------------------------------------------------ register --
    def register(self, doc: dict, tok: Tokenizer, permission: str) -> dict:
        """permission: 'never_train' (test) or 'eval_read_only' (validation)."""
        enc = tokenize_document(tok, doc)
        toks = enc["tokens"]
        entry = {
            "doc_id": doc["doc_id"],
            "split": doc["split"],
            "benchmark_id": doc.get("benchmark_id", ""),
            "version_tag": "TDES-BENCH-v1" if doc["split"] == "test" else "TDES-VAL-v1",
            "permission": permission,
            "never_train": permission == "never_train",
            "gradient_bearing_allowed": False,
            "text_hash": doc["text_hash"],
            "token_hash": hash_tokens(toks),
            "token_count": len(toks),
            "char_fingerprints": sorted(self._char_fp(doc["text"])),
            "token_fingerprints": sorted(self._token_fp(toks)),
        }
        self.entries[doc["doc_id"]] = entry
        self.text_hashes[doc["text_hash"]] = doc["doc_id"]
        for f in entry["char_fingerprints"]:
            self.char_fp[f] = doc["doc_id"]
        for f in entry["token_fingerprints"]:
            self.token_fp[f] = doc["doc_id"]
        return entry

    @staticmethod
    def _char_fp(text: str) -> Set[str]:
        t = normalize(text)
        return {hash_text(t[i:i + CHAR_WINDOW])[:16]
                for i in range(0, max(1, len(t) - CHAR_WINDOW + 1), CHAR_STRIDE)}

    @staticmethod
    def _token_fp(tokens: Sequence[int]) -> Set[str]:
        return {hash_tokens(tokens[i:i + TOKEN_NGRAM])[:16]
                for i in range(0, max(1, len(tokens) - TOKEN_NGRAM + 1))}

    # ---------------------------------------------------------------- scan --
    def scan_text(self, text: str) -> List[Tuple[str, str]]:
        """Sliding-window overlap test.  Returns [(doc_id, fingerprint), ...]."""
        t = normalize(text)
        hits = []
        for c in self.canaries:
            if c in t:
                hits.append(("canary:" + c[:20], "canary"))
        for i in range(0, max(1, len(t) - CHAR_WINDOW + 1)):
            f = hash_text(t[i:i + CHAR_WINDOW])[:16]
            d = self.char_fp.get(f)
            if d is not None:
                hits.append((d, f))
        return hits

    def scan_tokens(self, tokens: Sequence[int]) -> List[Tuple[str, str]]:
        hits = []
        for i in range(0, max(1, len(tokens) - TOKEN_NGRAM + 1)):
            f = hash_tokens(tokens[i:i + TOKEN_NGRAM])[:16]
            d = self.token_fp.get(f)
            if d is not None:
                hits.append((d, f))
        return hits

    def is_registered_text(self, text: str) -> bool:
        return hash_text(text) in self.text_hashes

    # ----------------------------------------------------------- firewall --
    def guard_batch(self, batch_id: str, step: int, tokens: Sequence[int],
                    loss_mask: Sequence[int], shard_ids: Iterable[str]) -> dict:
        """Serving-time firewall.  Only loss-bearing positions can leak."""
        bearing = [int(t) for t, m in zip(tokens, loss_mask) if m]
        hits = self.scan_tokens(bearing)
        ev = {"type": "batch_firewall_scan", "batch_id": batch_id, "step": step,
              "loss_bearing_tokens": len(bearing),
              "shard_ids": sorted(set(shard_ids)),
              "eval_overlap_hits": len(hits),
              "blocked": bool(hits),
              "hit_docs": sorted({h[0] for h in hits})}
        self.firewall_events.append(ev)
        return ev

    def record_access(self, who: str, doc_ids: List[str], purpose: str,
                      gradient_bearing: bool) -> dict:
        rec = {"who": who, "doc_ids": sorted(doc_ids), "purpose": purpose,
               "gradient_bearing": gradient_bearing,
               "permitted": (not gradient_bearing)}
        self.access_log.append(rec)
        return rec

    def log_admission(self, rec: dict) -> None:
        self.admission_log.append(rec)

    # ------------------------------------------------------------- persist --
    def write(self) -> str:
        doc = {
            "schema": "tdes-eval-registry/1",
            "canaries": self.canaries,
            "char_window": CHAR_WINDOW,
            "char_stride": CHAR_STRIDE,
            "token_ngram": TOKEN_NGRAM,
            "entry_count": len(self.entries),
            "entries": [self.entries[k] for k in sorted(self.entries)],
            "access_log": self.access_log,
            "firewall_events": self.firewall_events,
        }
        doc["registry_hash"] = hash_obj({k: v for k, v in doc.items()
                                         if k not in ("registry_hash", "access_log",
                                                      "firewall_events")})
        path = os.path.join(PATHS["manifests"], "eval_registry.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        return path

    def write_admission_report(self, extra: dict | None = None) -> dict:
        admitted = [r for r in self.admission_log if r["admitted"]]
        rejected = [r for r in self.admission_log if not r["admitted"]]
        reasons: Dict[str, int] = {}
        for r in rejected:
            for reason in r["reasons"]:
                key = reason.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        report = {
            "schema": "tdes-admission-report/1",
            "candidates": len(self.admission_log),
            "admitted": len(admitted),
            "rejected": len(rejected),
            "rejection_reason_counts": reasons,
            "records": self.admission_log,
        }
        if extra:
            report.update(extra)
        path = os.path.join(PATHS["manifests"], "admission_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        return report
