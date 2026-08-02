"""Throughput and packing-efficiency reporting.

Every headline number is emitted together with the raw counters it was derived
from, and each counter is independently recomputed from the ledgers, so a
reviewer can reconstruct the claim instead of trusting it.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .config import LANES, PATHS, SEQ_LEN, TOKENS_PER_STEP
from .hashing import hash_obj
from .ledger import LedgerSet

BRANCHES = ("main", "fork-a")


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def _phase_timings() -> dict:
    p = os.path.join(PATHS["reports"], "phase_timings.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_performance(log) -> int:
    log.section("PHASE 13 - PERFORMANCE")
    segments: List[dict] = []
    for br in BRANCHES:
        ls = LedgerSet(br)
        for r in ls["control"].read():
            if r["payload"].get("type") == "perf_segment":
                segments.append(r["payload"])

    agg = {"compute_seconds": 0.0, "loader_seconds": 0.0, "opus_seconds": 0.0,
           "wall_seconds": 0.0, "steps": 0, "positions": 0, "loss_tokens": 0,
           "real_tokens": 0, "accepted_seqs": 0, "candidates": 0}
    # a crashed segment and its resumed continuation both count once
    for s in segments:
        for k in agg:
            agg[k] += s.get(k, 0)

    # independent recomputation straight from the consumption ledgers.
    # "performed" = every batch the processes actually built and trained on,
    # including the steps lost to the crash; "committed" = the effective stream
    # after rollback.  The difference is the measurable cost of the crash.
    perf_pos = perf_loss = perf_real = perf_steps = 0
    ledger_positions = ledger_loss_tokens = ledger_real = steps_seen = 0
    lane_tokens = {l: 0 for l in LANES}
    for br in BRANCHES:
        ls = LedgerSet(br)
        for r in ls["consumption"].read():
            p = r["payload"]
            if p["type"] != "batch_served":
                continue
            perf_steps += 1
            perf_pos += p["n_positions"]
            perf_loss += p["n_loss_tokens"]
            perf_real += p["n_real_tokens"]
        for r in ls.effective_consumption():
            p = r["payload"]
            if p["type"] != "batch_served":
                continue
            steps_seen += 1
            ledger_positions += p["n_positions"]
            ledger_loss_tokens += p["n_loss_tokens"]
            ledger_real += p["n_real_tokens"]
            for lane, n in p["served_lanes"].items():
                lane_tokens[lane] += n * SEQ_LEN

    cache: Dict[str, float] = {"cache_hits": 0, "cache_misses": 0,
                               "shard_read_seconds": 0.0, "bytes_read": 0}
    rejects: Dict[str, int] = {l: 0 for l in LANES}
    for s in segments:
        for k in cache:
            cache[k] += s.get("shard_cache", {}).get(k, 0)
        for l, v in s.get("rejections_by_lane", {}).items():
            rejects[l] = rejects.get(l, 0) + v
    total_cache = cache["cache_hits"] + cache["cache_misses"]

    compute = agg["compute_seconds"]
    loader = agg["loader_seconds"] + agg["opus_seconds"]
    wall = agg["wall_seconds"]

    doc = {
        "schema": "tdes-performance/1",
        "raw_counters": {
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in agg.items()},
            "segments": len(segments),
            "segment_tags": [s["tag"] for s in segments],
        },
        "ledger_recomputation": {
            "performed_steps": perf_steps,
            "performed_positions": perf_pos,
            "performed_loss_bearing_tokens": perf_loss,
            "performed_real_tokens": perf_real,
            "committed_steps": steps_seen,
            "committed_positions": ledger_positions,
            "committed_loss_bearing_tokens": ledger_loss_tokens,
            "committed_real_tokens": ledger_real,
            "crash_wasted_positions": perf_pos - ledger_positions,
            "crash_wasted_steps": perf_steps - steps_seen,
            "tokens_per_step": TOKENS_PER_STEP,
            "positions_match_counters": perf_pos == agg["positions"],
            "loss_tokens_match_counters": perf_loss == agg["loss_tokens"],
            "note": ("performed = every batch built and trained on including the "
                     "steps lost to the crash; committed = effective stream after "
                     "rollback; counters come from the trainer, both totals are "
                     "re-derived from the ledger"),
        },
        "throughput": {
            "raw_tokens_per_sec": round(_safe_div(perf_pos, compute), 2),
            "useful_loss_bearing_tokens_per_sec": round(
                _safe_div(perf_loss, compute), 2),
            "accepted_tokens_per_sec_after_opus": round(
                _safe_div(perf_real, compute), 2),
            "committed_useful_tokens_per_sec": round(
                _safe_div(ledger_loss_tokens, compute), 2),
            "end_to_end_tokens_per_sec": round(_safe_div(perf_pos, wall), 2),
            "formula": ("performed_positions / compute_seconds; every input is in "
                        "raw_counters and ledger_recomputation"),
        },
        "packing": {
            "packing_utilization": round(_safe_div(perf_real, perf_pos), 6),
            "useful_token_ratio": round(_safe_div(perf_loss, perf_pos), 6),
            "padding_ratio": round(1.0 - _safe_div(perf_real, perf_pos), 6),
            "wasted_positions": perf_pos - perf_real,
            "context_only_positions": perf_real - perf_loss,
        },
        "loader": {
            "loader_seconds": round(loader, 6),
            "compute_seconds": round(compute, 6),
            "wall_seconds": round(wall, 6),
            "loader_wait_fraction": round(_safe_div(loader, compute + loader), 6),
            "gpu_idle_fraction_estimate": round(_safe_div(loader, wall), 6),
            "opus_scoring_seconds": round(agg["opus_seconds"], 6),
        },
        "shard_cache": {
            **cache,
            "cache_hit_rate": round(_safe_div(cache.get("cache_hits", 0),
                                              total_cache), 6),
        },
        "opus": {
            "candidates_scored": agg["candidates"],
            "sequences_accepted": agg["accepted_seqs"],
            "acceptance_rate": round(_safe_div(agg["accepted_seqs"],
                                               agg["candidates"]), 6),
            "rejections_by_lane": rejects,
        },
        "mixture_token_totals": lane_tokens,
        "phase_timings_seconds": _phase_timings(),
    }
    doc["performance_hash"] = hash_obj({k: v for k, v in doc.items()
                                        if k not in ("performance_hash",
                                                     "phase_timings_seconds")})
    with open(PATHS["performance"], "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)

    rec = doc["ledger_recomputation"]
    ok = rec["positions_match_counters"] and rec["loss_tokens_match_counters"] \
        and doc["throughput"]["useful_loss_bearing_tokens_per_sec"] > 0
    log.check(ok, "performance_measured",
              raw_tokens_per_sec=doc["throughput"]["raw_tokens_per_sec"],
              useful_tokens_per_sec=doc["throughput"][
                  "useful_loss_bearing_tokens_per_sec"],
              packing_utilization=doc["packing"]["packing_utilization"],
              useful_ratio=doc["packing"]["useful_token_ratio"],
              cache_hit_rate=doc["shard_cache"]["cache_hit_rate"])
    log.check(rec["positions_match_counters"] and rec["loss_tokens_match_counters"],
              "throughput_reconstructible_from_ledger",
              ledger_positions=rec["performed_positions"],
              counter_positions=doc["raw_counters"]["positions"],
              crash_wasted_positions=rec["crash_wasted_positions"])
    return 0 if ok else 1
