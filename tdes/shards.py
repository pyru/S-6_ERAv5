"""Immutable tokenized shards + manifests + the admission gate.

A shard is a sealed object: once written, its bytes are hashed and the file is
made read-only.  Any mutation produces a different ``content_hash`` and is
detected by :func:`verify_shard`.  Changing a shard means creating a *new*
shard with a new hash and a ``parent_shard_ids`` lineage entry.
"""
from __future__ import annotations

import array
import json
import os
import stat
import time
from typing import Dict, List, Tuple

from .config import (ALLOWED_LICENSE_TIERS, CLEANING_PIPELINE_VERSION, PATHS)
from .hashing import hash_obj, hash_tokens, merkle_root, sha256_hex
from .tokenizer import Tokenizer, tokenize_document

DOCS_PER_SHARD = 6


# --------------------------------------------------------------------------
# Admission gate (Session 3 + Session 4 contracts)
# --------------------------------------------------------------------------
def admission_check(doc: dict, registry) -> Tuple[bool, List[str]]:
    """Returns (admitted, reasons).  Reasons are recorded even on success."""
    reasons: List[str] = []
    if doc.get("split") != "train":
        reasons.append("not_training_split")
    if doc.get("holdout"):
        reasons.append("holdout_flag_set")
    if doc.get("license_tier") not in ALLOWED_LICENSE_TIERS:
        reasons.append("license_tier_not_admissible")
    if not doc.get("cleaning_pipeline_hash"):
        reasons.append("missing_cleaning_lineage")
    if doc.get("dedup_status") != "unique":
        reasons.append("duplicate_document")
    if doc.get("pii_status") != "clean":
        reasons.append("pii_detected")
    if not doc.get("lang_validated"):
        reasons.append("language_not_validated")
    hits = registry.scan_text(doc["text"])
    if hits:
        reasons.append("eval_contamination:" + ",".join(sorted(h[0] for h in hits)[:3]))
    if registry.is_registered_text(doc["text"]):
        reasons.append("registered_evaluation_document")
    return (len(reasons) == 0), reasons


def run_admission(docs: List[dict], registry, log=None) -> Tuple[List[dict], List[dict]]:
    admitted, rejected = [], []
    for d in docs:
        ok, reasons = admission_check(d, registry)
        rec = {"doc_id": d["doc_id"], "lane": d["lane"], "source_id": d["source_id"],
               "admitted": ok, "reasons": reasons, "text_hash": d["text_hash"]}
        (admitted if ok else rejected).append(d)
        if not ok and log is not None:
            log.event("admission_rejected", doc_id=d["doc_id"], reasons=",".join(reasons))
        registry.log_admission(rec)
    return admitted, rejected


# --------------------------------------------------------------------------
# Shard construction
# --------------------------------------------------------------------------
def build_shards(docs: List[dict], tok: Tokenizer, log=None) -> List[dict]:
    """Tokenize admitted documents and seal them into immutable shards."""
    by_lane: Dict[str, List[dict]] = {}
    for d in sorted(docs, key=lambda x: x["doc_id"]):
        by_lane.setdefault(d["lane"], []).append(d)

    manifests: List[dict] = []
    for lane in sorted(by_lane):
        group = by_lane[lane]
        for si in range((len(group) + DOCS_PER_SHARD - 1) // DOCS_PER_SHARD):
            chunk = group[si * DOCS_PER_SHARD:(si + 1) * DOCS_PER_SHARD]
            shard_id = f"shard-{lane}-{si:02d}"
            manifests.append(_seal_shard(shard_id, lane, chunk, tok))
            if log is not None:
                m = manifests[-1]
                log.event("shard_created", shard_id=shard_id, lane=lane,
                          docs=len(chunk), tokens=m["token_count"],
                          content_hash=m["content_hash"][:16])
    return manifests


def _seal_shard(shard_id: str, lane: str, docs: List[dict], tok: Tokenizer) -> dict:
    tokens: List[int] = []
    roles: List[str] = []
    loss: List[int] = []
    doc_records = []
    for d in docs:
        enc = tokenize_document(tok, d)
        start = len(tokens)
        tokens.extend(enc["tokens"])
        roles.extend(enc["roles"])
        loss.extend(enc["loss"])
        end = len(tokens)
        doc_records.append({
            "doc_id": d["doc_id"],
            "source_id": d["source_id"],
            "token_start": start,
            "token_end": end,
            "token_count": end - start,
            "lang": d["lang"],
            "script": d["script"],
            "license_tier": d["license_tier"],
            "provenance_tier": d["provenance_tier"],
            "capability_tags": d["capability_tags"],
            "text_hash": d["text_hash"],
            "token_hash": hash_tokens(enc["tokens"]),
            "structured": len(d["segments"]) > 1,
            "segments": [{"role": s["role"], "loss": s["loss"]} for s in d["segments"]],
        })

    bin_path = os.path.join(PATHS["shards"], shard_id + ".bin")
    _write_readonly(bin_path, array.array("I", tokens).tobytes())
    side_path = os.path.join(PATHS["shards"], shard_id + ".side.json")
    _write_readonly(side_path, json.dumps({"roles": roles, "loss": loss},
                                          separators=(",", ":")).encode("utf-8"))

    with open(bin_path, "rb") as fh:
        content_hash = sha256_hex(fh.read())

    manifest = {
        "shard_id": shard_id,
        "schema": "tdes-shard-manifest/1",
        "lane": lane,
        "capability_lane": lane,
        "source_ids": sorted({d["source_id"] for d in docs}),
        "doc_ids": [d["doc_id"] for d in docs],
        "documents": doc_records,
        "tokenizer_hash": tok.tokenizer_hash,
        "token_count": len(tokens),
        "loss_bearing_token_count": int(sum(loss)),
        "languages": sorted({d["lang"] for d in docs}),
        "scripts": sorted({d["script"] for d in docs}),
        "license_tiers": sorted({d["license_tier"] for d in docs}),
        "provenance_tiers": sorted({d["provenance_tier"] for d in docs}),
        "cleaning_pipeline_hash": sorted({d["cleaning_pipeline_hash"] for d in docs})[0],
        "cleaning_pipeline_version": CLEANING_PIPELINE_VERSION,
        "dedup_status": "unique",
        "contamination_status": "clean",
        "eval_overlap": False,
        "holdout": False,
        "never_train": False,
        "bin_path": os.path.relpath(bin_path, PATHS["art"]).replace("\\", "/"),
        "side_path": os.path.relpath(side_path, PATHS["art"]).replace("\\", "/"),
        "content_hash": content_hash,
        "token_stream_hash": hash_tokens(tokens),
        "parent_shard_ids": [],
        "sealed": True,
    }
    manifest["manifest_hash"] = hash_obj({k: v for k, v in manifest.items()
                                          if k != "manifest_hash"})
    mpath = os.path.join(PATHS["shard_manifests"], shard_id + ".json")
    _write_readonly(mpath, json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8"))
    return manifest


def _write_readonly(path: str, data: bytes) -> None:
    if os.path.exists(path):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    os.chmod(path, stat.S_IREAD)


# --------------------------------------------------------------------------
# Verification / loading
# --------------------------------------------------------------------------
class ShardStore:
    """Read-through cache over sealed shards, with real hit/miss accounting."""

    def __init__(self, manifests: List[dict], tokenizer_hash: str):
        self.manifests = {m["shard_id"]: m for m in manifests}
        self.tokenizer_hash = tokenizer_hash
        self._cache: Dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self.read_seconds = 0.0
        self.bytes_read = 0

    def get(self, shard_id: str) -> dict:
        if shard_id in self._cache:
            self.hits += 1
            return self._cache[shard_id]
        self.misses += 1
        t0 = time.perf_counter()
        m = self.manifests[shard_id]
        if m["tokenizer_hash"] != self.tokenizer_hash:
            raise ValueError(f"tokenizer hash mismatch for {shard_id}")
        bin_path = os.path.join(PATHS["art"], m["bin_path"])
        with open(bin_path, "rb") as fh:
            raw = fh.read()
        if sha256_hex(raw) != m["content_hash"]:
            raise ValueError(f"shard {shard_id} content hash mismatch (mutated shard)")
        arr = array.array("I")
        arr.frombytes(raw)
        with open(os.path.join(PATHS["art"], m["side_path"]), "r", encoding="utf-8") as fh:
            side = json.load(fh)
        entry = {"tokens": list(arr), "roles": side["roles"], "loss": side["loss"],
                 "manifest": m}
        self._cache[shard_id] = entry
        self.read_seconds += time.perf_counter() - t0
        self.bytes_read += len(raw)
        return entry

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "cache_hit_rate": (self.hits / total) if total else 0.0,
                "shard_read_seconds": self.read_seconds,
                "bytes_read": self.bytes_read,
                "mean_shard_read_ms": (self.read_seconds / self.misses * 1000.0)
                if self.misses else 0.0}


def verify_shard(manifest: dict) -> Tuple[bool, str]:
    bin_path = os.path.join(PATHS["art"], manifest["bin_path"])
    if not os.path.exists(bin_path):
        return False, "missing_file"
    with open(bin_path, "rb") as fh:
        h = sha256_hex(fh.read())
    if h != manifest["content_hash"]:
        return False, "content_hash_mismatch"
    recomputed = hash_obj({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if recomputed != manifest["manifest_hash"]:
        return False, "manifest_hash_mismatch"
    return True, "ok"


def write_manifest_index(manifests: List[dict], tok: Tokenizer,
                         admission_summary: dict) -> dict:
    leaves = [m["manifest_hash"] for m in sorted(manifests, key=lambda x: x["shard_id"])]
    index = {
        "schema": "tdes-manifest-index/1",
        "tokenizer_hash": tok.tokenizer_hash,
        "shard_count": len(manifests),
        "total_tokens": sum(m["token_count"] for m in manifests),
        "total_loss_bearing_tokens": sum(m["loss_bearing_token_count"] for m in manifests),
        "shards": [{"shard_id": m["shard_id"], "lane": m["lane"],
                    "tokens": m["token_count"], "content_hash": m["content_hash"],
                    "manifest_hash": m["manifest_hash"]}
                   for m in sorted(manifests, key=lambda x: x["shard_id"])],
        "merkle_root": merkle_root(leaves),
        "admission": admission_summary,
    }
    path = os.path.join(PATHS["manifests"], "manifest_index.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
    return index


def load_manifests() -> List[dict]:
    out = []
    d = PATHS["shard_manifests"]
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name), "r", encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out
