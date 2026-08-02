"""Evidence bundle generation.

Every value in evidence.json is read back out of the generated artifacts
(manifests, ledgers, checkpoints, reports, run_events.jsonl).  Nothing here
knows the "right" answer in advance: each requirement is a set of predicates
evaluated against files on disk, and the PASS/FAIL falls out of them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .config import CRASH_AT_STEP, PATHS, REPLAY_FROM, REPLAY_TO, TOTAL_STEPS
from .hashing import hash_obj
from .ledger import LedgerSet
from .runlog import read_events


def _load(*parts: str) -> Optional[dict]:
    path = os.path.join(*parts)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _rel(*parts: str) -> str:
    return os.path.relpath(os.path.join(*parts), PATHS["art"]).replace("\\", "/")


class Req:
    def __init__(self, key: str, title: str):
        self.key = key
        self.title = title
        self.checks: List[dict] = []
        self.evidence: List[dict] = []

    def check(self, name: str, passed: bool, observed: Any = None,
              expected: Any = None) -> "Req":
        self.checks.append({"check": name, "passed": bool(passed),
                            "observed": observed, "expected": expected})
        return self

    def ev(self, path: str, pointer: str = "", note: str = "") -> "Req":
        self.evidence.append({"artifact": path, "pointer": pointer, "note": note})
        return self

    def to_json(self) -> dict:
        ok = bool(self.checks) and all(c["passed"] for c in self.checks)
        return {"requirement": self.title, "key": self.key,
                "result": "PASS" if ok else "FAIL",
                "checks": self.checks, "evidence": self.evidence,
                "checks_passed": sum(1 for c in self.checks if c["passed"]),
                "checks_total": len(self.checks)}


def _log_index(events: List[dict]) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = {}
    for e in events:
        idx.setdefault(e["event"], []).append(e)
    return idx


def build_evidence(log) -> int:
    log.section("PHASE 14 - EVIDENCE BUNDLE")
    A = PATHS["art"]
    events = read_events()
    li = _log_index(events)

    def logged(name: str, level: str = "PASS") -> bool:
        return any(e["level"] == level for e in li.get(name, []))

    prep = _load(PATHS["manifests"], "prepare_summary.json") or {}
    index = _load(PATHS["manifests"], "manifest_index.json") or {}
    tokdoc = _load(PATHS["manifests"], "tokenizer.json") or {}
    packrep = _load(PATHS["manifests"], "packing_report.json") or {}
    admrep = _load(PATHS["manifests"], "admission_report.json") or {}
    audit = _load(PATHS["reports"], "audit.json") or {}
    resume = _load(PATHS["reports"], "resume_report.json") or {}
    replay = _load(PATHS["reports"], "replay_report.json") or {}
    fork = _load(PATHS["reports"], "fork_report.json") or {}
    learn = _load(PATHS["reports"], "learning_report.json") or {}
    mixc = _load(PATHS["reports"], "mixture_compliance.json") or {}
    perf = _load(PATHS["art"], "performance.json") or {}
    tests = _load(PATHS["reports"], "test_results.json") or {}

    reqs: List[Req] = []

    # ------------------------------------------------------- end-to-end run --
    r = Req("end_to_end", "End-to-end execution")
    required_events = ["shard_created", "manifests_validated", "eval_shard_blocked",
                       "mixture_compiled", "batch_packed", "opus_decisions_recorded",
                       "checkpoint_saved", "crash_simulated", "run_resumed",
                       "replay_step", "branch_forked", "audit_completed",
                       "performance_measured"]
    missing = [e for e in required_events if e not in li]
    r.check("all_phases_logged", not missing, observed=sorted(set(li) & set(
        required_events)), expected=required_events)
    r.check("no_FAIL_events", not [e for e in events if e["level"] == "FAIL"],
            observed=[e["event"] for e in events if e["level"] == "FAIL"], expected=[])
    r.ev(_rel(PATHS["run_log"]), "whole file", "complete execution log")
    r.ev("run_events.jsonl", "structured mirror of run.log")
    reqs.append(r)

    # -------------------------------------------------------------- tokenizer --
    r = Req("tokenizer_integrity", "Tokenizer integrity")
    r.check("tokenizer_hash_verified", logged("tokenizer_hash_verified"))
    r.check("roundtrip_lossless", logged("tokenizer_roundtrip_lossless"))
    r.check("indic_zero_width_preserved", logged("indic_zero_width_preserved"))
    r.check("all_shards_carry_frozen_hash",
            bool(audit.get("shards", {}).get("tokenizer_hash_match")),
            observed=audit.get("shards", {}).get("tokenizer_hashes_in_shards"),
            expected=[tokdoc.get("tokenizer_hash")])
    r.check("normalization_is_indic_safe",
            tokdoc.get("spec", {}).get("normalization", {}).get("form") == "NFC" and
            not tokdoc.get("spec", {}).get("normalization", {}).get("strip_zero_width"),
            observed=tokdoc.get("spec", {}).get("normalization"))
    r.ev(_rel(PATHS["manifests"], "tokenizer.json"), "tokenizer_hash")
    r.ev(_rel(PATHS["manifests"], "manifest_index.json"), "tokenizer_hash")
    reqs.append(r)

    # ------------------------------------------------------ shards/manifests --
    r = Req("shards_manifests", "Immutable shards and manifests")
    sh = audit.get("shards", {})
    r.check("all_shards_verify", sh.get("failed") == [], observed=sh.get("failed"))
    r.check("merkle_root_match", bool(sh.get("merkle_root_match")),
            observed=sh.get("merkle_root_recomputed"),
            expected=sh.get("merkle_root_in_index"))
    r.check("shard_count_matches_index",
            sh.get("shard_count") == index.get("shard_count"),
            observed=sh.get("shard_count"), expected=index.get("shard_count"))
    r.check("admission_gate_rejected_bad_docs",
            admrep.get("rejected", 0) >= 6,
            observed=admrep.get("rejection_reason_counts"))
    r.ev(_rel(PATHS["shard_manifests"]), "one manifest per shard")
    r.ev(_rel(PATHS["manifests"], "manifest_index.json"), "merkle_root")
    r.ev(_rel(PATHS["manifests"], "admission_report.json"), "rejection_reason_counts")
    reqs.append(r)

    # ----------------------------------------------------- packing correctness --
    r = Req("packing_correctness", "Packing, masks and batch correctness")
    rows = packrep.get("rows", [])
    r.check("zero_invariant_violations_across_all_policies",
            all(x["invariant_violations"] == 0 for x in rows),
            observed=sum(x["invariant_violations"] for x in rows), expected=0)
    r.check("policies_compared", len({x["policy"] for x in rows}) >= 6,
            observed=sorted({x["policy"] for x in rows}))
    r.check("structured_lanes_use_structure_safe_policy",
            all(x["structure_safe"] for x in rows if x["selected_for_training"]),
            observed=[x["policy"] for x in rows if x["selected_for_training"]])
    r.check("serve_time_invariants_enforced", logged("packing_invariants_verified"))
    r.check("no_batch_invariant_failures", not logged("batch_invariants", "FAIL"))
    r.ev(_rel(PATHS["manifests"], "packing_report.json"), "rows[].invariant_violations")
    r.ev(_rel(PATHS["packs"]), "packed windows with masks/positions/segments")
    reqs.append(r)

    # ------------------------------------------------------ mixture compliance --
    r = Req("mixture_compliance", "Mixture schedule, floors and curriculum")
    r.check("planned_vs_actual_within_tolerance", bool(mixc.get("compliant")),
            observed=mixc.get("max_abs_share_delta"), expected="<= 0.05")
    r.check("protected_floors_respected", bool(mixc.get("floors_respected")),
            observed=[f for f in mixc.get("protected_floors", [])
                      if not f["respected"]], expected=[])
    r.check("stages_compiled", logged("protected_floors_compiled"))
    r.ev(_rel(PATHS["manifests"], "mixture_schedule_main.json"), "per_step[].alloc_sequences")
    r.ev(_rel(PATHS["reports"], "mixture_compliance.json"), "rows[]")
    reqs.append(r)

    # -------------------------------------------------------------- OPUS trail --
    r = Req("opus_audit_trail", "OPUS acceptance, rejection, deferral, override")
    op = audit.get("opus", {})
    sc = op.get("status_counts", {})
    r.check("every_consumed_batch_has_a_decision",
            op.get("consumed_without_decision") == [],
            observed=op.get("consumed_without_decision"), expected=[])
    r.check("selection_matches_consumption",
            bool(op.get("selection_matches_consumption")))
    r.check("acceptances_recorded", sc.get("accepted", 0) > 0, observed=sc)
    r.check("rejections_retained", op.get("rejected_retained", 0) > 0,
            observed=op.get("rejected_retained"))
    r.check("deferrals_recorded",
            sum(v for k, v in sc.items() if k.startswith("deferred")) > 0, observed=sc)
    r.check("protected_floor_override_exercised",
            op.get("protected_floor_overrides", 0) > 0,
            observed=op.get("protected_floor_overrides"))
    r.check("scored_against_frozen_proxy_checkpoints",
            len(op.get("scoring_checkpoints", [])) > 1,
            observed=op.get("scoring_checkpoints"))
    r.ev("ledgers/main/opus.jsonl", "payload.status / rejection_reason")
    r.ev(_rel(PATHS["reports"], "audit.json"), "opus")
    reqs.append(r)

    # ------------------------------------------------------------- ledgers --
    r = Req("consumption_ledger", "Training consumption ledger")
    lm = audit.get("ledgers_main", {})
    r.check("hash_chains_verify", bool(lm.get("chains_ok")),
            observed=lm.get("chain_verification"))
    r.check("one_batch_record_per_step", bool(lm.get("contiguous")) and
            not lm.get("duplicate_steps"),
            observed=lm.get("steps_covered"), expected=[1, TOTAL_STEPS])
    r.check("microbatch_fanout_correct", bool(lm.get("microbatch_counts_ok")),
            observed=lm.get("microbatches_per_step_expected"))
    r.check("checkpoint_range_query_reconstructs_data",
            bool(audit.get("checkpoint_range_query", {}).get("ok")),
            observed=audit.get("checkpoint_range_query", {}).get("token_spans"))
    r.ev("ledgers/main/consumption.jsonl", "payload.token_spans / loss_mask_hash")
    r.ev(_rel(PATHS["reports"], "audit.json"), "ledgers_main")
    reqs.append(r)

    # ------------------------------------------------------- learning ledger --
    r = Req("learning_trace", "Learning ledger and token-level loss trace")
    r.check("learning_events_cover_every_consumed_pack",
            bool(learn.get("learning_events_cover_consumption")))
    r.check("token_level_trace_linked_to_source_data",
            bool(learn.get("token_trace_linked_to_consumption")),
            observed=learn.get("token_trace_records"))
    r.check("loss_decreased_over_the_run", bool(learn.get("loss_reduced")),
            observed=[learn.get("loss_start"), learn.get("loss_end")])
    r.check("per_shard_report_cards_generated",
            len(learn.get("shard_report_cards", [])) > 0,
            observed=len(learn.get("shard_report_cards", [])))
    r.check("repeated_pass_effect_measured",
            len(learn.get("repeated_pass_effect", [])) > 0,
            observed=learn.get("repeated_pass_effect"))
    r.ev("ledgers/main/learning.jsonl", "payload.loss_delta / shard_ids")
    r.ev("ledgers/main/token_trace.jsonl", "payload.ppl per token")
    r.ev(_rel(PATHS["reports"], "learning_report.json"), "shard_report_cards")
    reqs.append(r)

    # ------------------------------------------------------- crash / resume --
    r = Req("crash_recovery", "Crash recovery: no skipped or repeated batches")
    r.check("crash_was_a_real_process_exit", logged("crash_simulated", "EVENT"),
            observed=[e["fields"] for e in li.get("crash_simulated", [])])
    r.check("resume_next_batch_matched", bool(resume.get("matched")),
            observed=(resume.get("resumed_batch") or {}).get("batch_id"),
            expected=(resume.get("expected_batch") or {}).get("batch_id"))
    r.check("expected_batch_came_from_the_pre_crash_ledger",
            bool(resume.get("expected_batch")),
            observed=resume.get("orphan_steps"))
    r.check("effective_stream_contiguous", bool(resume.get("contiguous")),
            observed=resume.get("effective_steps"), expected=TOTAL_STEPS)
    r.check("no_duplicate_steps", resume.get("duplicate_steps") == [],
            observed=resume.get("duplicate_steps"), expected=[])
    r.check("ledger_chain_valid_at_checkpoint", bool(resume.get("ledger_chain_ok")))
    r.ev(_rel(PATHS["reports"], "resume_report.json"), "expected_batch vs resumed_batch")
    r.ev("ledgers/main/control.jsonl", "type=ledger_rollback")
    r.ev(_rel(PATHS["checkpoints"], "main"), "ledger_offsets + ledger_heads")
    reqs.append(r)

    # ------------------------------------------------------------- replay --
    r = Req("replay", "Replay of the historical data stream")
    rows = replay.get("rows", [])
    r.check("replay_all_hashes_match", bool(replay.get("all_match")),
            observed=[x["step"] for x in rows if not x["all_match"]], expected=[])
    r.check("batch_ids_match", all(x["batch_id_match"] for x in rows) and bool(rows))
    r.check("token_spans_match", all(x["token_spans_match"] for x in rows) and bool(rows))
    r.check("loss_mask_hashes_match",
            all(x["loss_mask_hash_match"] for x in rows) and bool(rows))
    r.check("interval_covered", [replay.get("interval")] == [[REPLAY_FROM, REPLAY_TO]],
            observed=replay.get("interval"), expected=[REPLAY_FROM, REPLAY_TO])
    r.check("digests_equal",
            replay.get("replay_digest") == replay.get("original_digest"),
            observed=replay.get("replay_digest"),
            expected=replay.get("original_digest"))
    r.ev(_rel(PATHS["reports"], "replay_report.json"), "rows[]")
    reqs.append(r)

    # --------------------------------------------------------------- fork --
    r = Req("fork", "Fork from an earlier checkpoint")
    r.check("fork_recorded", bool(fork.get("rows")))
    r.check("data_stream_diverged", bool(fork.get("all_diverged")),
            observed=[x["step"] for x in fork.get("rows", []) if not x["diverged"]],
            expected=[])
    r.check("model_state_inherited_from_parent",
            bool(fork.get("inherited_param_hash")),
            observed=fork.get("parent_checkpoint"))
    r.check("fork_ledger_chain_ok",
            bool(audit.get("ledgers_fork", {}).get("chains_ok")))
    r.ev(_rel(PATHS["reports"], "fork_report.json"), "rows[].fork_batch_id vs main")
    r.ev("ledgers/fork-a/control.jsonl", "type=branch_forked")
    reqs.append(r)

    # ---------------------------------------------------------- firewall --
    r = Req("eval_firewall", "Evaluation and validation firewall")
    fwa = audit.get("firewall", {})
    r.check("eval_shard_blocked_at_admission",
            len(fwa.get("admission_blocked_eval", [])) >= 2,
            observed=fwa.get("admission_blocked_eval"))
    r.check("zero_eval_overlap_in_served_batches",
            fwa.get("total_overlap_hits") == 0,
            observed=fwa.get("total_overlap_hits"), expected=0)
    r.check("no_eval_docs_in_training_stream",
            fwa.get("eval_docs_in_training_stream") == [],
            observed=fwa.get("eval_docs_in_training_stream"), expected=[])
    r.check("validation_never_gradient_bearing",
            fwa.get("validation_gradient_bearing") is False and
            fwa.get("validation_docs_in_training_stream") == [],
            observed=fwa.get("validation_eval_events"))
    r.check("every_batch_scanned",
            fwa.get("batches_scanned", 0) >= TOTAL_STEPS,
            observed=fwa.get("batches_scanned"), expected=f">= {TOTAL_STEPS}")
    r.ev(_rel(PATHS["manifests"], "eval_registry.json"), "entries[].never_train")
    r.ev("ledgers/main/firewall.jsonl", "eval_overlap_hits per batch")
    r.ev(_rel(PATHS["manifests"], "admission_report.json"), "blocked_eval_docs")
    reqs.append(r)

    # -------------------------------------------------------- throughput --
    r = Req("throughput", "Throughput and packing efficiency")
    tp = perf.get("throughput", {})
    pk = perf.get("packing", {})
    rc = perf.get("ledger_recomputation", {})
    r.check("useful_tokens_per_sec_reported",
            tp.get("useful_loss_bearing_tokens_per_sec", 0) > 0, observed=tp)
    r.check("counters_reconstructible_from_ledger",
            bool(rc.get("positions_match_counters")) and
            bool(rc.get("loss_tokens_match_counters")),
            observed=[rc.get("performed_positions"),
                      perf.get("raw_counters", {}).get("positions")])
    r.check("packing_utilization_reported",
            0.0 < pk.get("packing_utilization", 0) <= 1.0,
            observed=pk.get("packing_utilization"))
    r.check("packing_utilization_matches_ledger",
            abs(pk.get("packing_utilization", 0) -
                (rc.get("performed_real_tokens", 0) /
                 max(1, rc.get("performed_positions", 1)))) < 1e-6)
    r.ev(_rel(PATHS["performance"]), "throughput / packing / raw_counters")
    reqs.append(r)

    # ------------------------------------------------------------- tests --
    r = Req("tests", "Automated invariant tests")
    r.check("test_suite_executed", bool(tests), observed=tests.get("total"))
    r.check("all_tests_passed", tests.get("failures", 1) == 0 and
            tests.get("errors", 1) == 0 and tests.get("total", 0) > 0,
            observed={k: tests.get(k) for k in ("total", "failures", "errors")})
    r.ev(_rel(PATHS["reports"], "test_results.json"), "unittest summary")
    reqs.append(r)

    payload = [q.to_json() for q in reqs]
    overall = all(q["result"] == "PASS" for q in payload)
    doc = {
        "schema": "tdes-evidence/1",
        "generated_by": "tdes.evidence.build_evidence",
        "overall_result": "PASS" if overall else "FAIL",
        "requirements_passed": sum(1 for q in payload if q["result"] == "PASS"),
        "requirements_total": len(payload),
        "run": {
            "total_steps": TOTAL_STEPS,
            "crash_at_step": CRASH_AT_STEP,
            "replay_interval": [REPLAY_FROM, REPLAY_TO],
            "tokenizer_hash": prep.get("tokenizer_hash"),
            "shard_merkle_root": prep.get("merkle_root"),
            "mixture_schedule_hash": prep.get("mixture_schedule_hash"),
            "config_fingerprint": prep.get("config_fingerprint"),
        },
        "artifact_inventory": _inventory(),
        "requirements": payload,
    }
    doc["evidence_hash"] = hash_obj({k: v for k, v in doc.items()
                                     if k != "evidence_hash"})
    with open(PATHS["evidence_json"], "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    _write_md(doc)

    for q in payload:
        (log.ok if q["result"] == "PASS" else log.fail)(
            "evidence_" + q["key"], result=q["result"],
            checks=f"{q['checks_passed']}/{q['checks_total']}")
    log.check(overall, "evidence_bundle_generated",
              passed=doc["requirements_passed"], total=doc["requirements_total"],
              evidence_hash=doc["evidence_hash"][:16])
    return 0 if overall else 1


def _inventory() -> dict:
    inv = {}
    for root, _dirs, files in os.walk(PATHS["art"]):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, PATHS["art"]).replace("\\", "/")
            inv[rel] = os.path.getsize(p)
    return {"files": len(inv), "total_bytes": sum(inv.values()),
            "by_dir": _by_dir(inv)}


def _by_dir(inv: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k in inv:
        d = k.split("/")[0] if "/" in k else "."
        out[d] = out.get(d, 0) + 1
    return dict(sorted(out.items()))


# (requirement key, label as named in the assignment, evidence description)
ROW_ORDER = [
    ("tokenizer_integrity", "Tokenizer integrity", "Manifest record"),
    ("eval_firewall", "Evaluation firewall", "Blocked-shard event"),
    ("packing_correctness", "Packing correctness", "Packed-batch report"),
    ("mixture_compliance", "Mixture compliance", "Planned versus actual shares"),
    ("opus_audit_trail", "OPUS audit trail", "Candidate decision records"),
    ("crash_recovery", "Crash recovery", "Expected and resumed batch ids"),
    ("replay", "Replay", "Original and replay hashes"),
    ("learning_trace", "Learning trace", "Loss linked to source data"),
    ("throughput", "Throughput", "Performance report"),
]


def _write_md(doc: dict) -> None:
    by_key = {q["key"]: q for q in doc["requirements"]}
    lines = ["# TDES Evidence Summary", "",
             f"**Overall result: {doc['overall_result']}** "
             f"({doc['requirements_passed']}/{doc['requirements_total']} requirements)",
             "",
             f"- tokenizer hash: `{doc['run']['tokenizer_hash']}`",
             f"- shard merkle root: `{doc['run']['shard_merkle_root']}`",
             f"- mixture schedule hash: `{doc['run']['mixture_schedule_hash']}`",
             f"- steps: {doc['run']['total_steps']}, "
             f"crash at step {doc['run']['crash_at_step']}, "
             f"replay interval {doc['run']['replay_interval']}",
             f"- evidence hash: `{doc['evidence_hash']}`", "",
             "## Required summary", "",
             "| Requirement | Result | Evidence |",
             "| --- | --- | --- |"]
    for key, label, note in ROW_ORDER:
        q = by_key.get(key)
        if not q:
            lines.append(f"| {label} | **FAIL** | requirement not evaluated |")
            continue
        ev = "; ".join(f"`{e['artifact']}`" for e in q["evidence"][:2])
        lines.append(f"| {label} | **{q['result']}** | {note} - {ev} |")

    lines += ["", "## All requirements", "",
              "| # | Requirement | Result | Checks | Evidence |",
              "| --- | --- | --- | --- | --- |"]
    for i, q in enumerate(doc["requirements"], 1):
        ev = "; ".join(f"`{e['artifact']}`" for e in q["evidence"])
        lines.append(f"| {i} | {q['requirement']} | **{q['result']}** | "
                     f"{q['checks_passed']}/{q['checks_total']} | {ev} |")

    lines += ["", "## Check detail", ""]
    for q in doc["requirements"]:
        lines.append(f"### {q['requirement']} - {q['result']}")
        lines.append("")
        lines.append("| Check | Result | Observed | Expected |")
        lines.append("| --- | --- | --- | --- |")
        for c in q["checks"]:
            obs = _short(c["observed"])
            exp = _short(c["expected"])
            lines.append(f"| `{c['check']}` | {'PASS' if c['passed'] else 'FAIL'} "
                         f"| {obs} | {exp} |")
        lines.append("")
        for e in q["evidence"]:
            lines.append(f"- evidence: `{e['artifact']}`"
                         + (f" -> `{e['pointer']}`" if e["pointer"] else "")
                         + (f" ({e['note']})" if e["note"] else ""))
        lines.append("")
    lines += ["## Artifact inventory", "",
              f"{doc['artifact_inventory']['files']} files, "
              f"{doc['artifact_inventory']['total_bytes']} bytes", "",
              "| Directory | Files |", "| --- | --- |"]
    for d, n in doc["artifact_inventory"]["by_dir"].items():
        lines.append(f"| `{d}` | {n} |")
    lines.append("")
    with open(PATHS["evidence_md"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _short(v: Any, n: int = 90) -> str:
    if v is None:
        return "-"
    s = json.dumps(v, sort_keys=True, ensure_ascii=False) if not isinstance(v, str) else v
    s = s.replace("|", "\\|")
    return (s[:n] + "...") if len(s) > n else s
