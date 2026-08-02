"""The training stream: planned quotas -> candidates -> OPUS -> packed batch.

Determinism contract
--------------------
The batch served at ``(branch, step)`` is a function of exactly four things:

1. the branch config (id + seed + mixture override),
2. the compiled mixture schedule,
3. the immutable pack pools (which are a function of the sealed shards),
4. the frozen proxy checkpoint that OPUS scored against at that step.

All four are on disk, so an independent process can reconstruct any historical
batch without re-running the trainer.  That is what makes replay meaningful:
the replay path never reads the original batch, it *recomputes* it and only
then compares hashes.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List

from .config import (BranchConfig, DATALOADER_VERSION, GRAD_ACCUM, LANES,
                     MICRO_BATCH, OPUS_OVERGENERATION, PACKER_VERSION,
                     SEQS_PER_STEP, SEQ_LEN, WORLD_SIZE)
from .hashing import hash_obj, hash_tokens, stable_shuffle
from .mixture import MixtureSchedule


class TrainingStream:
    def __init__(self, branch: BranchConfig, mixture: MixtureSchedule,
                 pools: Dict[str, List[dict]], registry, opus, tokenizer_hash: str):
        self.branch = branch
        self.mixture = mixture
        self.registry = registry
        self.opus = opus
        self.tokenizer_hash = tokenizer_hash
        # deterministic per-branch pool permutation (immutable pool, shuffled view)
        self.pools = {lane: stable_shuffle(packs, "pool", branch.branch_id,
                                           branch.seed, lane)
                      for lane, packs in pools.items()}
        self.served_tokens: Dict[str, int] = {l: 0 for l in LANES}
        self.served_seqs: Dict[str, int] = {l: 0 for l in LANES}
        self.loader_seconds = 0.0
        self.candidates_scored = 0
        self.rejections_by_lane: Dict[str, int] = {l: 0 for l in LANES}

    # ------------------------------------------------------------- helpers --
    def _cand_count(self, step: int, lane: str) -> int:
        n = self.mixture.alloc(step)[lane]
        return 0 if n == 0 else int(math.ceil(n * OPUS_OVERGENERATION))

    def _cursor(self, step: int, lane: str) -> int:
        """Pure function of step: candidates consumed before this step."""
        return sum(self._cand_count(s, lane) for s in range(1, step))

    def _visible(self, lane: str, step: int) -> List[dict]:
        pool = self.pools.get(lane, [])
        if not pool:
            return []
        r = self.mixture.reserve_fraction(lane, step)
        keep = max(1, int(round(len(pool) * (1.0 - r))))
        return pool[:keep]

    def lane_deficit(self, lane: str, step: int) -> tuple:
        st = self.mixture.stage_for_step(step)
        floor = float(st["protected_floors"].get(lane, 0.0))
        target = float(st["effective_mixture"].get(lane, 0.0))
        total = sum(self.served_tokens.values())
        actual = (self.served_tokens[lane] / total) if total else target
        ref = floor if floor > 0 else target
        if ref <= 0:
            return 0.0, False
        deficit = max(0.0, (ref - actual) / ref)
        return min(1.0, deficit), (floor > 0 and actual < floor)

    # ---------------------------------------------------------- candidates --
    def candidates(self, step: int, lane: str) -> List[dict]:
        vis = self._visible(lane, step)
        if not vis:
            return []
        c = self._cand_count(step, lane)
        base = self._cursor(step, lane)
        st = self.mixture.stage_for_step(step)
        active = st["effective_mixture"].get(lane, 0.0) > 0.0
        out = []
        for slot in range(c):
            i = base + slot
            out.append({"pack": vis[i % len(vis)], "pass_no": i // len(vis),
                        "slot": slot, "stage_active": active})
        return out

    # --------------------------------------------------------------- batch --
    def build_batch(self, step: int, checkpoint_id: str, run_id: str) -> dict:
        t0 = time.perf_counter()
        st = self.mixture.stage_for_step(step)
        alloc = self.mixture.alloc(step)
        chosen: List[dict] = []
        decisions: List[dict] = []
        for lane in LANES:
            need = alloc[lane]
            if need <= 0:
                continue
            cands = self.candidates(step, lane)
            if not cands:
                continue
            deficit, protected = self.lane_deficit(lane, step)
            picked, decs = self.opus.select(step, st["stage"], lane, need, cands,
                                            deficit, protected)
            self.candidates_scored += len(decs)
            self.rejections_by_lane[lane] += sum(1 for d in decs
                                                 if d["status"] == "rejected")
            decisions.extend(decs)
            for c in picked:
                chosen.append({"lane": lane, **c})

        chosen.sort(key=lambda c: (c["lane"], c["decision"]["candidate_id"]))
        chosen = chosen[:SEQS_PER_STEP]
        if len(chosen) != SEQS_PER_STEP:
            raise RuntimeError(
                f"step {step}: loader produced {len(chosen)} sequences, "
                f"expected {SEQS_PER_STEP} - the batch shape must never be short")

        seqs = []
        idx = 0
        for accum in range(GRAD_ACCUM):
            for rank in range(WORLD_SIZE):
                for mb in range(MICRO_BATCH):
                    if idx >= len(chosen):
                        break
                    c = chosen[idx]
                    p = c["pack"]
                    seqs.append({
                        "seq_index": idx,
                        "rank": rank,
                        "accum": accum,
                        "micro_slot": mb,
                        "microbatch_id": f"mb-{step:05d}-r{rank}-a{accum}",
                        "lane": c["lane"],
                        "pack_id": p["pack_id"],
                        "pack_hash": p["pack_hash"],
                        "policy": p["policy"],
                        "candidate_id": c["decision"]["candidate_id"],
                        "pass_no": c["pass_no"],
                        "input_ids": p["input_ids"],
                        "loss_mask": p["loss_mask"],
                        "segment_ids": p["segment_ids"],
                        "position_ids": p["position_ids"],
                        "role_ids": p["role_ids"],
                        "target_loss": p["target_loss"],
                        "members": p["members"],
                        "n_loss_tokens": int(sum(p["loss_mask"])),
                        "n_real_tokens": p["n_real_tokens"],
                    })
                    idx += 1

        all_tokens: List[int] = []
        for s in seqs:
            all_tokens.extend(s["input_ids"])
        token_hash = hash_tokens(all_tokens)
        batch_core = {
            "branch": self.branch.branch_id,
            "global_step": step,
            "sequences": [{"pack_id": s["pack_id"], "pack_hash": s["pack_hash"],
                           "rank": s["rank"], "accum": s["accum"],
                           "seq_index": s["seq_index"]} for s in seqs],
            "token_hash": token_hash,
            "dataloader_version": DATALOADER_VERSION,
        }
        batch_hash = hash_obj(batch_core)
        batch_id = f"batch-{self.branch.branch_id}-{step:05d}-{batch_hash[:12]}"

        # serving-time evaluation firewall
        guard = self.registry.guard_batch(
            batch_id, step, all_tokens,
            [m for s in seqs for m in s["loss_mask"]],
            [m["shard_id"] for s in seqs for m in s["members"]])

        for s in seqs:
            self.served_tokens[s["lane"]] += SEQ_LEN
            self.served_seqs[s["lane"]] += 1

        microbatches = []
        for accum in range(GRAD_ACCUM):
            for rank in range(WORLD_SIZE):
                members = [s for s in seqs if s["rank"] == rank and s["accum"] == accum]
                if not members:
                    continue
                microbatches.append({
                    "microbatch_id": f"mb-{step:05d}-r{rank}-a{accum}",
                    "rank": rank, "accum": accum,
                    "seq_indices": [m["seq_index"] for m in members],
                })

        self.loader_seconds += time.perf_counter() - t0
        return {
            "batch_id": batch_id,
            "batch_hash": batch_hash,
            "token_hash": token_hash,
            "branch": self.branch.branch_id,
            "run_id": run_id,
            "global_step": step,
            "checkpoint_id": checkpoint_id,
            "curriculum_stage": st["stage"],
            "planned_alloc": alloc,
            "sequences": seqs,
            "microbatches": microbatches,
            "opus_decisions": decisions,
            "firewall": guard,
            "n_positions": len(all_tokens),
            "n_loss_tokens": int(sum(s["n_loss_tokens"] for s in seqs)),
            "n_real_tokens": int(sum(s["n_real_tokens"] for s in seqs)),
            "tokenizer_hash": self.tokenizer_hash,
            "dataloader_version": DATALOADER_VERSION,
            "packer_version": PACKER_VERSION,
        }

    # --------------------------------------------------------------- state --
    def state(self) -> dict:
        return {"served_tokens": dict(self.served_tokens),
                "served_seqs": dict(self.served_seqs),
                "candidates_scored": self.candidates_scored,
                "rejections_by_lane": dict(self.rejections_by_lane),
                "branch": self.branch.branch_id}

    def load_state(self, s: dict) -> None:
        self.served_tokens = {l: int(s["served_tokens"].get(l, 0)) for l in LANES}
        self.served_seqs = {l: int(s["served_seqs"].get(l, 0)) for l in LANES}
        self.candidates_scored = int(s.get("candidates_scored", 0))
        self.rejections_by_lane = {l: int(s.get("rejections_by_lane", {}).get(l, 0))
                                   for l in LANES}


def consumption_records(batch: dict) -> List[dict]:
    """Flatten a served batch into ledger payloads."""
    by_index = {s["seq_index"]: s for s in batch["sequences"]}
    out = [{
        "type": "batch_served",
        "run_id": batch["run_id"],
        "branch_id": batch["branch"],
        "global_step": batch["global_step"],
        "checkpoint_id": batch["checkpoint_id"],
        "batch_id": batch["batch_id"],
        "batch_hash": batch["batch_hash"],
        "token_hash": batch["token_hash"],
        "curriculum_stage": batch["curriculum_stage"],
        "planned_alloc": batch["planned_alloc"],
        "served_lanes": _lane_counts(batch),
        "n_positions": batch["n_positions"],
        "n_loss_tokens": batch["n_loss_tokens"],
        "n_real_tokens": batch["n_real_tokens"],
        "eval_firewall_hits": batch["firewall"]["eval_overlap_hits"],
        "tokenizer_hash": batch["tokenizer_hash"],
        "dataloader_version": batch["dataloader_version"],
        "packer_version": batch["packer_version"],
    }]
    for mb in batch["microbatches"]:
        seqs = [by_index[i] for i in mb["seq_indices"]]
        spans = []
        for s in seqs:
            for m in s["members"]:
                spans.append({"shard_id": m["shard_id"], "doc_id": m["doc_id"],
                              "src_start": m["src_start"], "src_end": m["src_end"],
                              "dst_start": m["dst_start"], "dst_end": m["dst_end"],
                              "pack_id": s["pack_id"], "token_hash": m["token_hash"]})
        out.append({
            "type": "microbatch_consumed",
            "run_id": batch["run_id"],
            "branch_id": batch["branch"],
            "global_step": batch["global_step"],
            "checkpoint_id": batch["checkpoint_id"],
            "batch_id": batch["batch_id"],
            "microbatch_id": mb["microbatch_id"],
            "rank": mb["rank"],
            "accum": mb["accum"],
            "packed_sample_ids": [s["pack_id"] for s in seqs],
            "shard_ids": sorted({sp["shard_id"] for sp in spans}),
            "token_spans": spans,
            "loss_mask_hash": hash_tokens([m for s in seqs for m in s["loss_mask"]]),
            "token_hash": hash_tokens([t for s in seqs for t in s["input_ids"]]),
            "attention_policy": "causal_block_diagonal_by_segment",
            "position_policy": "reset_per_segment",
            "mixture_lanes": sorted({s["lane"] for s in seqs}),
            "packing_policies": sorted({s["policy"] for s in seqs}),
            "curriculum_stage": batch["curriculum_stage"],
            "tokenizer_hash": batch["tokenizer_hash"],
            "dataloader_version": batch["dataloader_version"],
            "opus_decision_ids": [s["candidate_id"] for s in seqs],
            "repeated_pass_numbers": [s["pass_no"] for s in seqs],
            "n_loss_tokens": int(sum(s["n_loss_tokens"] for s in seqs)),
        })
    return out


def _lane_counts(batch: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in batch["sequences"]:
        out[s["lane"]] = out.get(s["lane"], 0) + 1
    return out


def batch_fingerprint(batch: dict) -> dict:
    """Small, comparable summary used by resume / replay proofs."""
    return {
        "batch_id": batch["batch_id"],
        "batch_hash": batch["batch_hash"],
        "token_hash": batch["token_hash"],
        "global_step": batch["global_step"],
        "pack_ids": [s["pack_id"] for s in batch["sequences"]],
        "pack_hashes": [s["pack_hash"] for s in batch["sequences"]],
        "loss_mask_hash": hash_tokens([m for s in batch["sequences"]
                                       for m in s["loss_mask"]]),
        "token_spans": [{"shard_id": m["shard_id"], "doc_id": m["doc_id"],
                         "src_start": m["src_start"], "src_end": m["src_end"]}
                        for s in batch["sequences"] for m in s["members"]],
        "lane_counts": _lane_counts(batch),
    }
