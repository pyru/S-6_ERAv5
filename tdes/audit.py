"""Audit: reconstruct the run from what is on disk and check every invariant.

Nothing in here reads in-memory state from the trainer.  It opens the ledgers,
manifests, checkpoints and reports, and re-derives the answers.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List

from .config import (CHECKPOINT_EVERY, GRAD_ACCUM, LANES, PATHS, SEQ_LEN,
                     TOTAL_STEPS, WORLD_SIZE)
from .hashing import hash_obj, merkle_root
from .ledger import LedgerSet
from .mixture import MixtureSchedule
from .shards import load_manifests, verify_shard
from .tokenizer import Tokenizer

MICRO_PER_STEP = WORLD_SIZE * GRAD_ACCUM


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _report(name: str, doc: dict) -> str:
    path = os.path.join(PATHS["reports"], name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return path


# --------------------------------------------------------------------------
def audit_shards() -> dict:
    manifests = load_manifests()
    tok = Tokenizer.load()
    index = _read_json(os.path.join(PATHS["manifests"], "manifest_index.json"))
    results = {m["shard_id"]: verify_shard(m) for m in manifests}
    bad = [k for k, v in results.items() if not v[0]]
    leaves = [m["manifest_hash"] for m in sorted(manifests, key=lambda x: x["shard_id"])]
    root = merkle_root(leaves)
    tok_uniform = sorted({m["tokenizer_hash"] for m in manifests})
    return {
        "shard_count": len(manifests),
        "verified": len(manifests) - len(bad),
        "failed": bad,
        "merkle_root_recomputed": root,
        "merkle_root_in_index": index["merkle_root"],
        "merkle_root_match": root == index["merkle_root"],
        "tokenizer_hashes_in_shards": tok_uniform,
        "frozen_tokenizer_hash": tok.tokenizer_hash,
        "tokenizer_hash_match": tok_uniform == [tok.tokenizer_hash],
        "total_tokens": sum(m["token_count"] for m in manifests),
        "immutability": "content_hash + read-only files; any edit changes the hash",
        "ok": (not bad) and root == index["merkle_root"] and
              tok_uniform == [tok.tokenizer_hash],
    }


def audit_ledgers(branch: str, expected_steps: int) -> dict:
    ls = LedgerSet(branch)
    chains = ls.verify()
    eff = ls.effective_consumption()
    served = [r["payload"] for r in eff if r["payload"]["type"] == "batch_served"]
    micro = [r["payload"] for r in eff if r["payload"]["type"] == "microbatch_consumed"]
    steps = [s["global_step"] for s in served]
    counts: Dict[int, int] = {}
    for m in micro:
        counts[m["global_step"]] = counts.get(m["global_step"], 0) + 1
    first = min(steps) if steps else 0
    contiguous = steps == list(range(first, first + len(steps)))
    dupes = sorted({s for s in steps if steps.count(s) > 1})
    micro_ok = all(v == MICRO_PER_STEP for v in counts.values())
    physical = ls["consumption"].read()
    rollbacks = [r["payload"] for r in ls["control"].read()
                 if r["payload"].get("type") == "ledger_rollback"]
    return {
        "branch": branch,
        "chain_verification": chains,
        "chains_ok": all(c["ok"] for c in chains.values()),
        "physical_consumption_records": len(physical),
        "effective_consumption_records": len(eff),
        "superseded_records": len(physical) - len(eff),
        "rollback_events": len(rollbacks),
        "steps_covered": [first, first + len(steps) - 1] if steps else [],
        "expected_steps": expected_steps,
        "contiguous": contiguous,
        "duplicate_steps": dupes,
        "microbatches_per_step_expected": MICRO_PER_STEP,
        "microbatch_counts_ok": micro_ok,
        "ok": (all(c["ok"] for c in chains.values()) and contiguous and not dupes
               and micro_ok and len(steps) == expected_steps),
    }


def audit_mixture(branch: str, override: Dict[str, float]) -> dict:
    ls = LedgerSet(branch)
    served = [r["payload"] for r in ls.effective_consumption()
              if r["payload"]["type"] == "batch_served"]
    mix = MixtureSchedule(override)
    actual_tokens = {l: 0 for l in LANES}
    planned_tokens = {l: 0 for l in LANES}
    per_stage: Dict[str, Dict[str, int]] = {}
    for s in served:
        stage = s["curriculum_stage"]
        per_stage.setdefault(stage, {l: 0 for l in LANES})
        for lane, n in s["served_lanes"].items():
            actual_tokens[lane] += n * SEQ_LEN
            per_stage[stage][lane] += n * SEQ_LEN
        for lane, n in s["planned_alloc"].items():
            planned_tokens[lane] += n * SEQ_LEN
    total = sum(actual_tokens.values()) or 1
    ptotal = sum(planned_tokens.values()) or 1
    rows = []
    for lane in LANES:
        rows.append({"lane": lane,
                     "planned_share": round(planned_tokens[lane] / ptotal, 6),
                     "actual_share": round(actual_tokens[lane] / total, 6),
                     "planned_tokens": planned_tokens[lane],
                     "actual_tokens": actual_tokens[lane],
                     "abs_delta": round(abs(actual_tokens[lane] / total -
                                            planned_tokens[lane] / ptotal), 6)})
    floors = []
    for stage_name, tokens in per_stage.items():
        st = next(s for s in mix.stages if s["stage"] == stage_name)
        tot = sum(tokens.values()) or 1
        for lane, floor in st["protected_floors"].items():
            share = tokens[lane] / tot
            floors.append({"stage": stage_name, "lane": lane, "floor": floor,
                           "actual_share": round(share, 6),
                           "respected": share >= floor - 0.02})
    max_delta = max(r["abs_delta"] for r in rows) if rows else 0.0
    return {
        "branch": branch,
        "rows": rows,
        "max_abs_share_delta": max_delta,
        "tolerance": 0.05,
        "compliant": max_delta <= 0.05,
        "protected_floors": floors,
        "floors_respected": all(f["respected"] for f in floors),
        "ok": max_delta <= 0.05 and all(f["respected"] for f in floors),
    }


def audit_opus(branch: str) -> dict:
    ls = LedgerSet(branch)
    decisions = [r["payload"] for r in ls["opus"].read()]
    consumed_packs = set()
    consumed_cands = set()
    for r in ls.effective_consumption():
        p = r["payload"]
        if p["type"] == "microbatch_consumed":
            consumed_packs.update(p["packed_sample_ids"])
            consumed_cands.update(p["opus_decision_ids"])
    by_status: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    for d in decisions:
        by_status[d["status"]] = by_status.get(d["status"], 0) + 1
        if d["rejection_reason"]:
            reasons[d["rejection_reason"]] = reasons.get(d["rejection_reason"], 0) + 1
    known = {d["candidate_id"] for d in decisions}
    missing = sorted(consumed_cands - known)
    selected = {d["candidate_id"] for d in decisions if d["selected"]}
    overrides = [d for d in decisions if d["protected_floor_override"]]
    rejected_kept = [d for d in decisions if d["status"] == "rejected"]
    return {
        "branch": branch,
        "candidate_records": len(decisions),
        "status_counts": by_status,
        "rejection_reason_counts": reasons,
        "protected_floor_overrides": len(overrides),
        "override_examples": overrides[:3],
        "rejected_retained": len(rejected_kept),
        "consumed_candidates": len(consumed_cands),
        "consumed_without_decision": missing,
        "selection_matches_consumption": selected == consumed_cands,
        "proxy_versions": sorted({d["proxy_version"] for d in decisions}),
        "scoring_checkpoints": sorted({d["scoring_checkpoint_id"] for d in decisions}),
        "has_all_four_outcomes": all(
            any(k.startswith(s) for k in by_status)
            for s in ("accepted", "rejected", "deferred")),
        "ok": (not missing) and selected == consumed_cands and len(decisions) > 0,
    }


def audit_firewall(branch: str) -> dict:
    ls = LedgerSet(branch)
    fw = [r["payload"] for r in ls["firewall"].read()]
    reg = _read_json(os.path.join(PATHS["manifests"], "eval_registry.json"))
    adm = _read_json(os.path.join(PATHS["manifests"], "admission_report.json"))
    eval_docs = {e["doc_id"] for e in reg["entries"] if e["never_train"]}
    val_docs = {e["doc_id"] for e in reg["entries"] if not e["never_train"]}
    consumed_docs = set()
    for r in ls.effective_consumption():
        p = r["payload"]
        if p["type"] == "microbatch_consumed":
            consumed_docs.update(sp["doc_id"] for sp in p["token_spans"])
    blocked = [r for r in adm["records"] if not r["admitted"]]
    blocked_eval = [r for r in blocked
                    if any(x.startswith("eval_contamination") or
                           x == "registered_evaluation_document" for x in r["reasons"])]
    val_events = [r["payload"] for r in ls["control"].read()
                  if r["payload"].get("type") == "validation_eval"]
    return {
        "branch": branch,
        "registered_test_docs": len(eval_docs),
        "registered_validation_docs": len(val_docs),
        "batches_scanned": len(fw),
        "total_overlap_hits": sum(f["eval_overlap_hits"] for f in fw),
        "blocked_batches": sum(1 for f in fw if f["blocked"]),
        "eval_docs_in_training_stream": sorted(eval_docs & consumed_docs),
        "validation_docs_in_training_stream": sorted(val_docs & consumed_docs),
        "admission_blocked_total": len(blocked),
        "admission_blocked_eval": [r["doc_id"] for r in blocked_eval],
        "admission_rejection_reasons": adm["rejection_reason_counts"],
        "validation_eval_events": len(val_events),
        "validation_gradient_bearing": any(v["gradient_bearing"] for v in val_events),
        "ok": (sum(f["eval_overlap_hits"] for f in fw) == 0
               and not (eval_docs & consumed_docs)
               and not (val_docs & consumed_docs)
               and len(blocked_eval) >= 2
               and not any(v["gradient_bearing"] for v in val_events)),
    }


def audit_learning(branch: str) -> dict:
    ls = LedgerSet(branch)
    events = [r["payload"] for r in ls["learning"].read()]
    traces = [r["payload"] for r in ls["token_trace"].read()]
    consumed_packs = set()
    for r in ls.effective_consumption():
        p = r["payload"]
        if p["type"] == "microbatch_consumed":
            consumed_packs.update(p["packed_sample_ids"])

    by_shard: Dict[str, dict] = {}
    by_lane: Dict[str, dict] = {}
    by_pass: Dict[int, dict] = {}
    for e in events:
        for sid in e["shard_ids"]:
            a = by_shard.setdefault(sid, {"exposures": 0, "loss_before": 0.0,
                                          "loss_after": 0.0, "delta": 0.0,
                                          "loss_tokens": 0, "grad_norm": 0.0,
                                          "lanes": set(), "passes": set()})
            a["exposures"] += 1
            a["loss_before"] += e["mean_token_loss_before"]
            a["loss_after"] += e["mean_token_loss_after"]
            a["delta"] += e["loss_delta"]
            a["loss_tokens"] += e["n_loss_tokens"]
            a["grad_norm"] += e["grad_norm"]
            a["lanes"].add(e["lane"])
            a["passes"].add(e["repeated_pass_number"])
        b = by_lane.setdefault(e["lane"], {"exposures": 0, "delta": 0.0,
                                           "loss_before": 0.0, "loss_tokens": 0})
        b["exposures"] += 1
        b["delta"] += e["loss_delta"]
        b["loss_before"] += e["mean_token_loss_before"]
        b["loss_tokens"] += e["n_loss_tokens"]
        c = by_pass.setdefault(e["repeated_pass_number"],
                               {"exposures": 0, "delta": 0.0})
        c["exposures"] += 1
        c["delta"] += e["loss_delta"]

    shard_rows = []
    for sid, a in sorted(by_shard.items()):
        n = a["exposures"]
        mean_delta = a["delta"] / n
        shard_rows.append({
            "shard_id": sid,
            "lanes": sorted(a["lanes"]),
            "exposures": n,
            "repeated_passes": sorted(a["passes"]),
            "mean_token_loss_before": round(a["loss_before"] / n, 6),
            "mean_token_loss_after": round(a["loss_after"] / n, 6),
            "mean_loss_delta": round(mean_delta, 8),
            "mean_perplexity_before": round(math.exp(min(20.0, a["loss_before"] / n)), 4),
            "loss_bearing_tokens": a["loss_tokens"],
            "mean_grad_norm": round(a["grad_norm"] / n, 6),
            "usefulness": ("useful" if mean_delta > 1e-4 else
                           "harmful" if mean_delta < -1e-4 else "neutral"),
        })
    lane_rows = [{"lane": k, "exposures": v["exposures"],
                  "mean_loss_delta": round(v["delta"] / v["exposures"], 8),
                  "mean_loss_before": round(v["loss_before"] / v["exposures"], 6),
                  "loss_bearing_tokens": v["loss_tokens"]}
                 for k, v in sorted(by_lane.items())]
    pass_rows = [{"repeated_pass_number": k, "exposures": v["exposures"],
                  "mean_loss_delta": round(v["delta"] / v["exposures"], 8)}
                 for k, v in sorted(by_pass.items())]

    first = [e for e in events if e["global_step"] <= 4]
    last = [e for e in events if e["global_step"] >= max(
        (x["global_step"] for x in events), default=0) - 3]
    loss_start = sum(e["mean_token_loss_before"] for e in first) / max(1, len(first))
    loss_end = sum(e["mean_token_loss_after"] for e in last) / max(1, len(last))

    traced_packs = {t["pack_id"] for t in traces}
    linked = traced_packs.issubset(consumed_packs) and bool(traced_packs)
    event_packs = {e["pack_id"] for e in events}
    doc = {
        "branch": branch,
        "learning_events": len(events),
        "token_trace_records": len(traces),
        "token_trace_packs": sorted(traced_packs),
        "token_trace_linked_to_consumption": linked,
        "learning_events_cover_consumption": event_packs == consumed_packs,
        "loss_start": round(loss_start, 6),
        "loss_end": round(loss_end, 6),
        "loss_reduced": loss_end < loss_start,
        "shard_report_cards": shard_rows,
        "lane_rows": lane_rows,
        "repeated_pass_effect": pass_rows,
        "classification_counts": _counts([e["classification"] for e in events]),
    }
    doc["ok"] = bool(events) and linked and doc["learning_events_cover_consumption"] \
        and doc["loss_reduced"]
    return doc


def _counts(xs) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def audit_checkpoint_range(branch: str, lo: int, hi: int) -> dict:
    """Which data influenced the model between two checkpoints?"""
    ls = LedgerSet(branch)
    shards: Dict[str, int] = {}
    docs: Dict[str, int] = {}
    lanes: Dict[str, int] = {}
    spans = 0
    for r in ls.effective_consumption():
        p = r["payload"]
        if p["type"] != "microbatch_consumed" or not (lo < p["global_step"] <= hi):
            continue
        for sp in p["token_spans"]:
            shards[sp["shard_id"]] = shards.get(sp["shard_id"], 0) + \
                (sp["src_end"] - sp["src_start"])
            docs[sp["doc_id"]] = docs.get(sp["doc_id"], 0) + 1
            spans += 1
        for lane in p["mixture_lanes"]:
            lanes[lane] = lanes.get(lane, 0) + 1
    return {"branch": branch, "checkpoint_range": [lo, hi],
            "token_spans": spans, "shards": dict(sorted(shards.items())),
            "documents": len(docs), "lanes": dict(sorted(lanes.items())),
            "question": f"which shards influenced the model between step {lo} and {hi}",
            "ok": spans > 0}


# --------------------------------------------------------------------------
def run_audit(log) -> int:
    log.section("PHASE 12 - AUDIT")
    shards = audit_shards()
    log.check(shards["ok"], "shard_manifest_audit",
              shards=shards["shard_count"], merkle_match=shards["merkle_root_match"],
              tokenizer_match=shards["tokenizer_hash_match"])

    led = audit_ledgers("main", TOTAL_STEPS)
    log.check(led["ok"], "consumption_ledger_audit",
              effective=led["effective_consumption_records"],
              superseded=led["superseded_records"],
              contiguous=led["contiguous"], duplicates=led["duplicate_steps"])

    fled = audit_ledgers("fork-a", 8)
    log.check(fled["chains_ok"], "fork_ledger_chain_verified",
              records=fled["effective_consumption_records"])

    mixc = audit_mixture("main", {})
    log.check(mixc["ok"], "mixture_compliance",
              max_delta=mixc["max_abs_share_delta"],
              floors_respected=mixc["floors_respected"])

    opus = audit_opus("main")
    log.check(opus["ok"], "opus_audit_trail",
              candidates=opus["candidate_records"],
              statuses=opus["status_counts"],
              protected_floor_overrides=opus["protected_floor_overrides"])

    fw = audit_firewall("main")
    log.check(fw["ok"], "evaluation_firewall_audit",
              scanned=fw["batches_scanned"], hits=fw["total_overlap_hits"],
              blocked_at_admission=len(fw["admission_blocked_eval"]),
              validation_gradient_bearing=fw["validation_gradient_bearing"])

    learn = audit_learning("main")
    log.check(learn["ok"], "learning_trace_audit",
              events=learn["learning_events"],
              token_trace=learn["token_trace_records"],
              loss_start=learn["loss_start"], loss_end=learn["loss_end"])

    rng = audit_checkpoint_range("main", 24, 32)
    log.check(rng["ok"], "checkpoint_range_reconstructed",
              range="24..32", shards=len(rng["shards"]), spans=rng["token_spans"])

    doc = {
        "schema": "tdes-audit/1",
        "shards": shards, "ledgers_main": led, "ledgers_fork": fled,
        "mixture": mixc, "opus": opus, "firewall": fw, "learning": learn,
        "checkpoint_range_query": rng,
    }
    doc["ok"] = all(doc[k].get("ok", True) for k in
                    ("shards", "ledgers_main", "mixture", "opus", "firewall",
                     "learning", "checkpoint_range_query"))
    doc["audit_hash"] = hash_obj({k: v for k, v in doc.items() if k != "audit_hash"})
    _report("audit.json", doc)
    _report("learning_report.json", learn)
    _report("mixture_compliance.json", mixc)
    log.check(doc["ok"], "audit_completed", audit_hash=doc["audit_hash"][:16])
    return 0 if doc["ok"] else 1
