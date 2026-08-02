"""Append-only, hash-chained ledgers.

Every ledger is a JSONL file where record *n* commits to record *n-1*:

    record_hash = sha256(prev_hash || sha256(canonical_json(payload)))

so truncation, reordering or in-place edits are detectable.  A checkpoint
stores ``(offset, head_hash)`` for each ledger; on resume the chain is verified
up to that offset and everything after it is explicitly *superseded* by an
appended rollback record rather than deleted.  The ledger therefore stays
append-only while the *effective* consumption sequence is still exactly one
record per global step.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

from .config import PATHS
from .hashing import ZERO, chain_hash

LEDGERS = ("consumption", "opus", "learning", "firewall", "control", "token_trace")


def branch_dir(branch: str) -> str:
    d = os.path.join(PATHS["ledgers"], branch)
    os.makedirs(d, exist_ok=True)
    return d


class Ledger:
    def __init__(self, branch: str, name: str):
        self.branch = branch
        self.name = name
        self.path = os.path.join(branch_dir(branch), name + ".jsonl")
        self.head = ZERO
        self.count = 0
        if os.path.exists(self.path):
            for rec in self.read():
                self.head = rec["record_hash"]
                self.count += 1
        else:
            open(self.path, "w", encoding="utf-8").close()

    def append(self, payload: dict) -> dict:
        h = chain_hash(self.head, payload)
        rec = {"seq": self.count, "branch": self.branch, "ledger": self.name,
               "prev_hash": self.head, "record_hash": h, "payload": payload}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
        self.head = h
        self.count += 1
        return rec

    def read(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    # ------------------------------------------------------------- verify --
    def verify_chain(self, upto: Optional[int] = None) -> Dict[str, object]:
        recs = self.read()
        if upto is not None:
            recs = recs[:upto]
        prev = ZERO
        for i, r in enumerate(recs):
            if r["seq"] != i:
                return {"ok": False, "reason": f"seq_gap_at_{i}", "head": prev,
                        "count": i}
            if r["prev_hash"] != prev:
                return {"ok": False, "reason": f"prev_hash_break_at_{i}", "head": prev,
                        "count": i}
            if chain_hash(prev, r["payload"]) != r["record_hash"]:
                return {"ok": False, "reason": f"record_hash_break_at_{i}",
                        "head": prev, "count": i}
            prev = r["record_hash"]
        return {"ok": True, "reason": "ok", "head": prev, "count": len(recs)}

    def head_at(self, offset: int) -> str:
        recs = self.read()[:offset]
        return recs[-1]["record_hash"] if recs else ZERO


class LedgerSet:
    """The six ledgers of one branch, plus supersede bookkeeping."""

    def __init__(self, branch: str):
        self.branch = branch
        self.l: Dict[str, Ledger] = {n: Ledger(branch, n) for n in LEDGERS}

    def __getitem__(self, name: str) -> Ledger:
        return self.l[name]

    def offsets(self) -> Dict[str, int]:
        return {n: self.l[n].count for n in LEDGERS}

    def heads(self) -> Dict[str, str]:
        return {n: self.l[n].head for n in LEDGERS}

    def verify(self) -> Dict[str, dict]:
        return {n: self.l[n].verify_chain() for n in LEDGERS}

    # ---------------------------------------------------------- supersede --
    def supersede_after(self, offsets: Dict[str, int], reason: str) -> dict:
        """Mark every record beyond the checkpointed offset as superseded."""
        dropped = {}
        for n in LEDGERS:
            cur = self.l[n].count
            keep = offsets.get(n, cur)
            dropped[n] = max(0, cur - keep)
        rec = self.l["control"].append({
            "type": "ledger_rollback",
            "reason": reason,
            "committed_offsets": offsets,
            "superseded_counts": dropped,
        })
        return {"record": rec, "superseded": dropped}

    def effective_consumption(self) -> List[dict]:
        """Consumption records still in force after all rollbacks."""
        rollbacks = [r["payload"] for r in self.l["control"].read()
                     if r["payload"].get("type") == "ledger_rollback"]
        recs = self.l["consumption"].read()
        alive = [True] * len(recs)
        for rb in rollbacks:
            keep = rb["committed_offsets"].get("consumption", len(recs))
            # records appended before this rollback but beyond the committed
            # offset are dropped from the effective stream
            n_at_rollback = keep + rb["superseded_counts"].get("consumption", 0)
            for i in range(keep, min(n_at_rollback, len(recs))):
                alive[i] = False
        return [r for r, a in zip(recs, alive) if a]


def consumption_index(records: Iterable[dict]) -> Dict[int, List[dict]]:
    idx: Dict[int, List[dict]] = {}
    for r in records:
        p = r["payload"]
        if p.get("type") == "microbatch_consumed":
            idx.setdefault(p["global_step"], []).append(p)
    return idx
