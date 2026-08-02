"""Tests that re-verify the generated run from the artifacts on disk.

Each case declares the artifacts it needs and skips if they are not there yet,
so the file is safe to run standalone.  ``run_demo.py`` runs the suite twice:
once before the evidence bundle is written (its result feeds the bundle) and
once afterwards, when the evidence checks below also become live.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tdes.config import (CHECKPOINT_EVERY, CRASH_AT_STEP, GRAD_ACCUM, PATHS,
                         REPLAY_FROM, REPLAY_TO, SEQ_LEN, TOTAL_STEPS, WORLD_SIZE)
from tdes.hashing import chain_hash, hash_obj, hash_tokens, merkle_root, sha256_hex
from tdes.ledger import LedgerSet
from tdes.packing import build_pack_pools, load_pack_pools, verify_pack
from tdes.shards import ShardStore, load_manifests, verify_shard
from tdes.tokenizer import Tokenizer

MICRO_PER_STEP = WORLD_SIZE * GRAD_ACCUM

AUDIT = os.path.join(PATHS["reports"], "audit.json")
PERF = PATHS["performance"]
EVIDENCE = PATHS["evidence_json"]


def _load(*parts):
    with open(os.path.join(*parts), "r", encoding="utf-8") as fh:
        return json.load(fh)


class ArtifactCase(unittest.TestCase):
    """Skips (rather than fails) until the artifacts it needs exist."""
    REQUIRES = (AUDIT,)

    def setUp(self):
        for p in self.REQUIRES:
            if not os.path.exists(p):
                self.skipTest("artifact not generated yet: " + os.path.basename(p))


# --------------------------------------------------------------------------
class TestShardIntegrity(ArtifactCase):
    def setUp(self):
        super().setUp()
        self.manifests = load_manifests()
        self.tok = Tokenizer.load()

    def test_every_shard_verifies(self):
        for m in self.manifests:
            ok, why = verify_shard(m)
            self.assertTrue(ok, f"{m['shard_id']}: {why}")

    def test_shards_are_read_only_on_disk(self):
        for m in self.manifests[:3]:
            p = os.path.join(PATHS["art"], m["bin_path"])
            self.assertFalse(os.access(p, os.W_OK), f"{m['shard_id']} is writable")

    def test_mutating_a_shard_changes_its_content_hash(self):
        m = self.manifests[0]
        with open(os.path.join(PATHS["art"], m["bin_path"]), "rb") as fh:
            raw = bytearray(fh.read())
        self.assertEqual(sha256_hex(bytes(raw)), m["content_hash"])
        raw[0] ^= 0xFF
        self.assertNotEqual(sha256_hex(bytes(raw)), m["content_hash"])

    def test_manifest_hash_covers_every_field(self):
        m = dict(self.manifests[0])
        h = m.pop("manifest_hash")
        self.assertEqual(hash_obj(m), h)
        m["token_count"] = m["token_count"] + 1
        self.assertNotEqual(hash_obj(m), h)

    def test_merkle_root_matches_index(self):
        index = _load(PATHS["manifests"], "manifest_index.json")
        leaves = [m["manifest_hash"]
                  for m in sorted(self.manifests, key=lambda x: x["shard_id"])]
        self.assertEqual(merkle_root(leaves), index["merkle_root"])

    def test_all_shards_carry_the_frozen_tokenizer_hash(self):
        for m in self.manifests:
            self.assertEqual(m["tokenizer_hash"], self.tok.tokenizer_hash)

    def test_shard_documents_reconstruct_from_token_spans(self):
        store = ShardStore(self.manifests, self.tok.tokenizer_hash)
        for m in self.manifests[:4]:
            entry = store.get(m["shard_id"])
            for d in m["documents"]:
                span = entry["tokens"][d["token_start"]:d["token_end"]]
                self.assertEqual(hash_tokens(span), d["token_hash"])


class TestPacksAreReproducible(ArtifactCase):
    def test_rebuilding_pools_reproduces_every_pack_hash(self):
        manifests = load_manifests()
        tok = Tokenizer.load()
        store = ShardStore(manifests, tok.tokenizer_hash)
        rebuilt = build_pack_pools(store, manifests)
        stored = load_pack_pools(sorted(rebuilt))
        for lane in rebuilt:
            self.assertEqual([p["pack_hash"] for p in rebuilt[lane]],
                             [p["pack_hash"] for p in stored[lane]], lane)

    def test_every_stored_pack_satisfies_invariants(self):
        pools = load_pack_pools(sorted({m["lane"] for m in load_manifests()}))
        for lane, packs in pools.items():
            for p in packs:
                self.assertEqual(verify_pack(p), [], f"{lane}/{p['pack_id']}")

    def test_validation_packs_are_marked_never_train(self):
        path = os.path.join(PATHS["packs"], "_validation.jsonl")
        with open(path, "r", encoding="utf-8") as fh:
            packs = [json.loads(l) for l in fh if l.strip()]
        self.assertTrue(packs)
        for p in packs:
            self.assertTrue(p["never_train"])
            self.assertEqual(verify_pack(p), [])


class TestLedgers(ArtifactCase):
    def setUp(self):
        super().setUp()
        self.ls = LedgerSet("main")

    def test_every_chain_verifies(self):
        for name, res in self.ls.verify().items():
            self.assertTrue(res["ok"], f"{name}: {res['reason']}")

    def test_tampering_with_a_payload_breaks_the_chain(self):
        recs = self.ls["consumption"].read()
        r = recs[1]
        self.assertEqual(chain_hash(r["prev_hash"], r["payload"]), r["record_hash"])
        bad = dict(r["payload"])
        bad["global_step"] = 999
        self.assertNotEqual(chain_hash(r["prev_hash"], bad), r["record_hash"])

    def test_effective_stream_has_one_batch_per_step_no_gaps(self):
        served = [r["payload"] for r in self.ls.effective_consumption()
                  if r["payload"]["type"] == "batch_served"]
        steps = [s["global_step"] for s in served]
        self.assertEqual(steps, list(range(1, TOTAL_STEPS + 1)))
        self.assertEqual(len(set(steps)), len(steps))

    def test_crash_left_superseded_records_behind(self):
        physical = self.ls["consumption"].read()
        effective = self.ls.effective_consumption()
        self.assertGreater(len(physical), len(effective),
                           "the crash should have left uncommitted records")
        rollbacks = [r["payload"] for r in self.ls["control"].read()
                     if r["payload"].get("type") == "ledger_rollback"]
        self.assertTrue(rollbacks)

    def test_microbatch_fanout_matches_topology(self):
        counts = {}
        for r in self.ls.effective_consumption():
            p = r["payload"]
            if p["type"] == "microbatch_consumed":
                counts[p["global_step"]] = counts.get(p["global_step"], 0) + 1
        self.assertTrue(counts)
        for step, n in counts.items():
            self.assertEqual(n, MICRO_PER_STEP, f"step {step}")

    def test_token_spans_point_at_real_shard_content(self):
        manifests = load_manifests()
        tok = Tokenizer.load()
        store = ShardStore(manifests, tok.tokenizer_hash)
        checked = 0
        for r in self.ls.effective_consumption()[:40]:
            p = r["payload"]
            if p["type"] != "microbatch_consumed":
                continue
            for sp in p["token_spans"][:4]:
                entry = store.get(sp["shard_id"])
                span = entry["tokens"][sp["src_start"]:sp["src_end"]]
                self.assertEqual(hash_tokens(span), sp["token_hash"])
                checked += 1
        self.assertGreater(checked, 0)

    def test_every_consumed_pack_has_a_learning_event(self):
        packs = set()
        for r in self.ls.effective_consumption():
            p = r["payload"]
            if p["type"] == "microbatch_consumed":
                packs.update(p["packed_sample_ids"])
        learned = {r["payload"]["pack_id"] for r in self.ls["learning"].read()}
        self.assertEqual(packs, learned)

    def test_token_trace_rows_link_back_to_consumed_packs(self):
        traces = [r["payload"] for r in self.ls["token_trace"].read()]
        self.assertTrue(traces)
        packs = set()
        for r in self.ls.effective_consumption():
            p = r["payload"]
            if p["type"] == "microbatch_consumed":
                packs.update(p["packed_sample_ids"])
        for t in traces[:200]:
            self.assertIn(t["pack_id"], packs)
            self.assertGreater(t["ppl"], 0.0)


class TestFirewall(ArtifactCase):
    def test_no_eval_or_validation_document_entered_training(self):
        reg = _load(PATHS["manifests"], "eval_registry.json")
        held = {e["doc_id"] for e in reg["entries"]}
        consumed = set()
        for r in LedgerSet("main").effective_consumption():
            p = r["payload"]
            if p["type"] == "microbatch_consumed":
                consumed.update(sp["doc_id"] for sp in p["token_spans"])
        self.assertEqual(held & consumed, set())

    def test_every_batch_was_scanned_and_clean(self):
        fw = [r["payload"] for r in LedgerSet("main")["firewall"].read()]
        self.assertGreaterEqual(len(fw), TOTAL_STEPS)
        self.assertEqual(sum(f["eval_overlap_hits"] for f in fw), 0)
        self.assertFalse(any(f["blocked"] for f in fw))

    def test_admission_blocked_the_poisoned_documents(self):
        adm = _load(PATHS["manifests"], "admission_report.json")
        reasons = adm["rejection_reason_counts"]
        for expected in ("license_tier_not_admissible", "missing_cleaning_lineage",
                         "duplicate_document", "pii_detected",
                         "language_not_validated", "eval_contamination"):
            self.assertIn(expected, reasons)

    def test_validation_was_read_but_never_gradient_bearing(self):
        events = [r["payload"] for r in LedgerSet("main")["control"].read()
                  if r["payload"].get("type") == "validation_eval"]
        self.assertTrue(events)
        for e in events:
            self.assertFalse(e["gradient_bearing"])
            self.assertTrue(e["access_permitted"])


class TestOpus(ArtifactCase):
    def test_all_four_outcomes_are_present(self):
        decisions = [r["payload"] for r in LedgerSet("main")["opus"].read()]
        statuses = {d["status"] for d in decisions}
        self.assertIn("accepted", statuses)
        self.assertIn("rejected", statuses)
        self.assertTrue(any(s.startswith("deferred") for s in statuses))
        self.assertTrue(any(d["protected_floor_override"] for d in decisions))

    def test_every_served_sequence_has_a_decision_record(self):
        ls = LedgerSet("main")
        decisions = {r["payload"]["candidate_id"]: r["payload"]
                     for r in ls["opus"].read()}
        for r in ls.effective_consumption():
            p = r["payload"]
            if p["type"] != "microbatch_consumed":
                continue
            for cid in p["opus_decision_ids"]:
                self.assertIn(cid, decisions)
                self.assertTrue(decisions[cid]["selected"])

    def test_rejections_carry_a_reason_and_are_retained(self):
        decisions = [r["payload"] for r in LedgerSet("main")["opus"].read()]
        rejected = [d for d in decisions if d["status"] == "rejected"]
        self.assertTrue(rejected)
        for d in rejected:
            self.assertTrue(d["rejection_reason"])
            self.assertFalse(d["selected"])

    def test_scores_are_bound_to_a_frozen_proxy_checkpoint(self):
        decisions = [r["payload"] for r in LedgerSet("main")["opus"].read()]
        ckpts = {d["scoring_checkpoint_id"] for d in decisions}
        self.assertGreater(len(ckpts), 1)
        for d in decisions[:50]:
            self.assertTrue(d["proxy_version"])


class TestCheckpointCrashResumeReplayFork(ArtifactCase):
    def test_checkpoints_bind_model_state_to_ledger_offsets(self):
        from tdes import checkpoint as ck
        ls = LedgerSet("main")
        for step in ck.list_checkpoints("main"):
            doc = ck.load_checkpoint("main", step)
            self.assertIn("consumption", doc["ledger_offsets"])
            self.assertEqual(doc["next_step"], step + 1)
            self.assertEqual(
                ls["consumption"].head_at(doc["ledger_offsets"]["consumption"]),
                doc["ledger_heads"]["consumption"])

    def test_checkpoint_tamper_is_detected(self):
        from tdes import checkpoint as ck
        doc = ck.load_checkpoint("main", CHECKPOINT_EVERY)
        mutated = dict(doc)
        mutated["next_step"] = 999
        recomputed = hash_obj({k: v for k, v in mutated.items()
                               if k not in ("params", "momentum", "checkpoint_hash")})
        self.assertNotEqual(recomputed, doc["checkpoint_hash"])

    def test_resume_served_exactly_the_expected_batch(self):
        rep = _load(PATHS["reports"], "resume_report.json")
        self.assertTrue(rep["matched"])
        self.assertEqual(rep["expected_batch"]["batch_id"],
                         rep["resumed_batch"]["batch_id"])
        self.assertEqual(rep["expected_batch"]["token_hash"],
                         rep["resumed_batch"]["token_hash"])
        self.assertEqual(rep["resumed_batch"]["global_step"], rep["next_step"])
        self.assertIn(CRASH_AT_STEP, rep["orphan_steps"])

    def test_resume_did_not_skip_or_repeat(self):
        rep = _load(PATHS["reports"], "resume_report.json")
        self.assertTrue(rep["contiguous"])
        self.assertEqual(rep["duplicate_steps"], [])
        self.assertEqual(rep["effective_steps"], TOTAL_STEPS)

    def test_replay_reconstructed_identical_batches(self):
        rep = _load(PATHS["reports"], "replay_report.json")
        self.assertEqual(rep["interval"], [REPLAY_FROM, REPLAY_TO])
        self.assertTrue(rep["rows"])
        self.assertEqual(rep["replay_digest"], rep["original_digest"])
        for row in rep["rows"]:
            self.assertTrue(row["all_match"], row)
            self.assertEqual(row["original_batch_hash"], row["replay_batch_hash"])
            self.assertTrue(row["token_spans_match"])

    def test_fork_shares_model_state_but_diverges_in_data(self):
        from tdes import checkpoint as ck
        rep = _load(PATHS["reports"], "fork_report.json")
        parent = ck.load_checkpoint(rep["parent_branch"], rep["fork_step"])
        self.assertEqual(parent["param_hash"], rep["inherited_param_hash"])
        self.assertTrue(rep["all_diverged"])
        for row in rep["rows"]:
            self.assertNotEqual(row["fork_batch_id"], row["main_batch_id"])

    def test_fork_branch_has_its_own_ledger_chain(self):
        for name, res in LedgerSet("fork-a").verify().items():
            self.assertTrue(res["ok"], f"fork-a/{name}: {res['reason']}")


class TestMixture(ArtifactCase):
    def test_actual_shares_track_the_plan(self):
        mixc = _load(PATHS["reports"], "mixture_compliance.json")
        self.assertTrue(mixc["compliant"], mixc["rows"])
        for f in mixc["protected_floors"]:
            self.assertTrue(f["respected"], f)

    def test_served_tokens_match_the_ledger(self):
        served = [r["payload"] for r in LedgerSet("main").effective_consumption()
                  if r["payload"]["type"] == "batch_served"]
        for s in served:
            self.assertEqual(sum(s["served_lanes"].values()) * SEQ_LEN,
                             s["n_positions"])


class TestPerformance(ArtifactCase):
    REQUIRES = (AUDIT, PERF)

    def test_performance_numbers_are_reconstructible(self):
        perf = _load(PATHS["art"], "performance.json")
        rc = perf["ledger_recomputation"]
        self.assertTrue(rc["positions_match_counters"])
        self.assertTrue(rc["loss_tokens_match_counters"])
        self.assertAlmostEqual(
            perf["packing"]["packing_utilization"],
            rc["performed_real_tokens"] / rc["performed_positions"], places=6)
        self.assertAlmostEqual(
            perf["throughput"]["useful_loss_bearing_tokens_per_sec"],
            rc["performed_loss_bearing_tokens"] /
            perf["raw_counters"]["compute_seconds"], delta=1.0)

    def test_crash_waste_is_accounted_for(self):
        perf = _load(PATHS["art"], "performance.json")
        rc = perf["ledger_recomputation"]
        self.assertGreater(rc["crash_wasted_steps"], 0)
        self.assertEqual(rc["crash_wasted_positions"],
                         rc["performed_positions"] - rc["committed_positions"])


class TestEvidenceMatchesArtifacts(ArtifactCase):
    REQUIRES = (AUDIT, PERF, EVIDENCE)

    def test_every_evidence_artifact_exists(self):
        ev = _load(PATHS["art"], "evidence.json")
        for req in ev["requirements"]:
            for e in req["evidence"]:
                p = os.path.join(PATHS["art"], e["artifact"])
                self.assertTrue(os.path.exists(p), e["artifact"])

    def test_result_follows_from_the_checks(self):
        ev = _load(PATHS["art"], "evidence.json")
        for req in ev["requirements"]:
            expected = "PASS" if all(c["passed"] for c in req["checks"]) else "FAIL"
            self.assertEqual(req["result"], expected, req["key"])
        self.assertEqual(
            ev["overall_result"],
            "PASS" if all(r["result"] == "PASS" for r in ev["requirements"])
            else "FAIL")

    def test_evidence_hash_covers_the_document(self):
        ev = _load(PATHS["art"], "evidence.json")
        h = ev.pop("evidence_hash")
        self.assertEqual(hash_obj(ev), h)

    def test_evidence_md_lists_every_required_row(self):
        with open(PATHS["evidence_md"], "r", encoding="utf-8") as fh:
            md = fh.read()
        for row in ("Tokenizer integrity", "Evaluation and validation firewall",
                    "Packing, masks and batch correctness",
                    "Mixture schedule, floors and curriculum",
                    "OPUS acceptance, rejection, deferral, override",
                    "Crash recovery", "Replay of the historical data stream",
                    "Learning ledger and token-level loss trace",
                    "Throughput and packing efficiency"):
            self.assertIn(row, md, row)

    def test_run_log_contains_the_required_event_sequence(self):
        with open(PATHS["run_log"], "r", encoding="utf-8") as fh:
            text = fh.read()
        for marker in ("shard_created", "manifests_validated", "eval_shard_blocked",
                       "mixture_compiled", "batch_packed", "opus_decisions_recorded",
                       "checkpoint_saved", "crash_simulated", "run_resumed",
                       "replay_hash_matched", "branch_forked", "audit_completed",
                       "performance_measured",
                       "[PASS] tokenizer_hash_verified",
                       "[PASS] eval_shard_blocked",
                       "[PASS] checkpoint_saved",
                       "[PASS] resume_next_batch_matched",
                       "[PASS] replay_hash_matched"):
            self.assertIn(marker, text, marker)

    def test_no_fail_events_were_logged(self):
        from tdes.runlog import read_events
        fails = [e for e in read_events() if e["level"] == "FAIL"]
        self.assertEqual(fails, [])


if __name__ == "__main__":
    unittest.main()
