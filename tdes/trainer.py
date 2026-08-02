"""Training loop: consumption, learning, checkpoints, crash, resume, replay, fork."""
from __future__ import annotations

import json
import math
import os
import time
from typing import List, Optional

import numpy as np

from . import checkpoint as ckpt
from .config import (BranchConfig, CHECKPOINT_EVERY, D_FF, D_MODEL, GRAD_CLIP,
                     INIT_SEED, LEARNING_RATE, MOMENTUM, PATHS, SEQ_LEN,
                     TOKENS_PER_STEP, VOCAB_SIZE)
from .dataloader import TrainingStream, consumption_records
from .hashing import hash_obj, hash_tokens
from .ledger import LedgerSet
from .mixture import MixtureSchedule
from .model import TinyModel
from .opus import OpusSelector
from .packing import load_pack_pools, verify_pack
from .registry import EvalRegistry
from .shards import ShardStore, load_manifests
from .tokenizer import Tokenizer

TRACE_EVERY = 8


class CrashInjected(Exception):
    pass


# --------------------------------------------------------------------------
# environment reconstruction (everything comes from disk)
# --------------------------------------------------------------------------
def load_registry(tok: Tokenizer) -> EvalRegistry:
    with open(os.path.join(PATHS["corpus"], "corpus.json"), "r", encoding="utf-8") as fh:
        corpus = json.load(fh)
    reg = EvalRegistry()
    for d in corpus["test"]:
        reg.register(d, tok, "never_train")
    for d in corpus["validation"]:
        reg.register(d, tok, "eval_read_only")
    return reg


def build_env(branch: BranchConfig) -> dict:
    tok = Tokenizer.load()
    manifests = load_manifests()
    store = ShardStore(manifests, tok.tokenizer_hash)
    pools = load_pack_pools(sorted({m["lane"] for m in manifests}))
    reg = load_registry(tok)
    mixture = MixtureSchedule(branch.mixture_override)
    return {"tok": tok, "manifests": manifests, "store": store, "pools": pools,
            "registry": reg, "mixture": mixture}


def new_model() -> TinyModel:
    return TinyModel(vocab=VOCAB_SIZE, d_model=D_MODEL, d_ff=D_FF,
                     max_pos=SEQ_LEN + 8, seed=INIT_SEED)


def proxy_step_for(step: int) -> int:
    return ((step - 1) // CHECKPOINT_EVERY) * CHECKPOINT_EVERY


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------
class Trainer:
    def __init__(self, branch: BranchConfig, run_id: str, log, env: Optional[dict] = None):
        self.branch = branch
        self.run_id = run_id
        self.log = log
        self.env = env or build_env(branch)
        self.model = new_model()
        self.ledgers = LedgerSet(branch.branch_id)
        self.opus = OpusSelector(new_model(), ckpt.ckpt_id(branch.branch_id, 0))
        self.stream = TrainingStream(branch, self.env["mixture"], self.env["pools"],
                                     self.env["registry"], self.opus,
                                     self.env["tok"].tokenizer_hash)
        self.current_ckpt = ckpt.ckpt_id(branch.branch_id, 0)
        self._proxy_bound = -1
        self.perf = {"compute_seconds": 0.0, "loader_seconds": 0.0,
                     "opus_seconds": 0.0, "steps": 0, "positions": 0,
                     "loss_tokens": 0, "real_tokens": 0, "accepted_seqs": 0,
                     "candidates": 0, "wall_seconds": 0.0}
        self._t_start = time.perf_counter()

    # ---------------------------------------------------------------- setup --
    def bootstrap(self) -> dict:
        """Save the step-0 checkpoint that anchors the branch."""
        doc = ckpt.save_checkpoint(
            self.branch.branch_id, self.run_id, 0, 1, self.model,
            self.stream.state(), self.ledgers,
            {"parent_checkpoint": "", "parent_branch": self.branch.parent_branch,
             "fork_step": self.branch.fork_step,
             "mixture_schedule_hash": self.env["mixture"].to_json()["schedule_hash"],
             "tokenizer_hash": self.env["tok"].tokenizer_hash})
        self.ledgers["control"].append({
            "type": "checkpoint_saved", "checkpoint_id": doc["checkpoint_id"],
            "global_step": 0, "next_step": 1,
            "ledger_offsets": doc["ledger_offsets"],
            "param_hash": doc["param_hash"],
            "checkpoint_hash": doc["checkpoint_hash"]})
        self.log.ok("checkpoint_saved", checkpoint_id=doc["checkpoint_id"], step=0)
        return doc

    def restore(self, step: int) -> dict:
        doc = ckpt.load_checkpoint(self.branch.branch_id, step)
        ckpt.restore_model(self.model, doc)
        self.stream.load_state(doc["stream_state"])
        self.current_ckpt = doc["checkpoint_id"]
        return doc

    def restore_from(self, branch_id: str, step: int) -> dict:
        """Fork: take model + data position from another branch's checkpoint."""
        doc = ckpt.load_checkpoint(branch_id, step)
        ckpt.restore_model(self.model, doc)
        return doc

    def _bind_proxy(self, step: int) -> None:
        ps = proxy_step_for(step)
        if ps == self._proxy_bound:
            return
        branch = self.branch.branch_id
        path = ckpt.ckpt_path(branch, ps)
        if not os.path.exists(path) and self.branch.parent_branch:
            branch = self.branch.parent_branch
        doc = ckpt.load_checkpoint(branch, ps)
        m = new_model()
        ckpt.restore_model(m, doc)
        self.opus.rebind(m, doc["checkpoint_id"])
        self._proxy_bound = ps
        self.log.info("opus_proxy_bound", step=step,
                      scoring_checkpoint_id=doc["checkpoint_id"],
                      proxy_version=self.opus.proxy_version)

    # ----------------------------------------------------------------- step --
    def serve(self, step: int) -> dict:
        self._bind_proxy(step)
        t0 = time.perf_counter()
        batch = self.stream.build_batch(step, self.current_ckpt, self.run_id)
        self.perf["opus_seconds"] += time.perf_counter() - t0
        return batch

    def consume(self, step: int, trace: bool = True) -> dict:
        batch = self.serve(step)

        # --- serving-time firewall: a blocked batch must never reach the optimizer
        if batch["firewall"]["blocked"]:
            self.log.fail("eval_data_entered_batch", step=step,
                          batch_id=batch["batch_id"],
                          hits=batch["firewall"]["hit_docs"])
            raise RuntimeError("evaluation data reached a loss-bearing batch")

        # --- provenance: every served sequence is re-read from the sealed shard
        #     through the loader cache, so the cache/latency numbers in
        #     performance.json come from real reads and the token spans in the
        #     ledger are proven to point at real shard content
        store = self.env["store"]
        bad = []
        for s in batch["sequences"]:
            for m in s["members"]:
                entry = store.get(m["shard_id"])
                span = entry["tokens"][m["src_start"]:m["src_end"]]
                if hash_tokens(span) != m["token_hash"]:
                    bad.append(m["sample_id"])
        if bad:
            self.log.fail("token_span_provenance", step=step, spans=bad[:5])
            raise RuntimeError("served token span does not match its sealed shard")

        # --- pack invariants are re-checked at serve time, not just at build time
        violations = []
        for s in batch["sequences"]:
            violations.extend(verify_pack({
                "seq_len": SEQ_LEN, "input_ids": s["input_ids"],
                "loss_mask": s["loss_mask"], "segment_ids": s["segment_ids"],
                "position_ids": s["position_ids"], "role_ids": s["role_ids"],
                "target_loss": s["target_loss"], "members": s["members"],
                "n_real_tokens": s["n_real_tokens"],
                "n_pad_tokens": SEQ_LEN - s["n_real_tokens"],
                "token_hash": hash_tokens(s["input_ids"])}))
        if violations:
            self.log.fail("batch_invariants", step=step, violations=violations[:5])
            raise RuntimeError("packed batch violated invariants: %s" % violations[:3])

        # --- learning: loss before and after the update, per sequence
        t0 = time.perf_counter()
        pre = []
        for s in batch["sequences"]:
            l, tl, n = self.model.sequence_loss(s["input_ids"], s["position_ids"],
                                                s["segment_ids"], s["loss_mask"])
            pre.append((l, tl, n))
        grads = self.model.zero_grads()
        nseq = len(batch["sequences"])
        for s in batch["sequences"]:
            self.model.accumulate(grads, s["input_ids"], s["position_ids"],
                                  s["segment_ids"], s["loss_mask"], 1.0 / nseq)
        grad_norm = self.model.step(grads, LEARNING_RATE, MOMENTUM, GRAD_CLIP)
        post = []
        for s in batch["sequences"]:
            l, tl, n = self.model.sequence_loss(s["input_ids"], s["position_ids"],
                                                s["segment_ids"], s["loss_mask"])
            post.append((l, tl, n))
        self.perf["compute_seconds"] += time.perf_counter() - t0

        # --- ledgers
        for rec in consumption_records(batch):
            self.ledgers["consumption"].append(rec)
        for d in batch["opus_decisions"]:
            self.ledgers["opus"].append({"type": "opus_decision", **d})
        self.ledgers["firewall"].append({**batch["firewall"],
                                         "batch_id": batch["batch_id"]})

        mean_pre = float(np.mean([p[0] for p in pre]))
        mean_post = float(np.mean([p[0] for p in post]))
        for i, s in enumerate(batch["sequences"]):
            delta = pre[i][0] - post[i][0]
            cls = ("useful" if delta > 1e-4 else
                   "harmful" if delta < -1e-4 else "neutral")
            hi = _high_ppl_clusters(s, pre[i][1], self.env["tok"])
            self.ledgers["learning"].append({
                "type": "learning_event",
                "run_id": self.run_id,
                "branch_id": self.branch.branch_id,
                "global_step": step,
                "batch_id": batch["batch_id"],
                "pack_id": s["pack_id"],
                "lane": s["lane"],
                "curriculum_stage": batch["curriculum_stage"],
                "shard_ids": sorted({m["shard_id"] for m in s["members"]}),
                "doc_ids": sorted({m["doc_id"] for m in s["members"]}),
                "candidate_id": s["candidate_id"],
                "repeated_pass_number": s["pass_no"],
                "n_loss_tokens": pre[i][2],
                "mean_token_loss_before": round(pre[i][0], 8),
                "mean_token_loss_after": round(post[i][0], 8),
                "loss_delta": round(delta, 8),
                "mean_token_ppl_before": round(math.exp(min(20.0, pre[i][0])), 6),
                "grad_norm": round(grad_norm, 8),
                "model_tokens_seen": (step - 1) * TOKENS_PER_STEP,
                "checkpoint_before": self.current_ckpt,
                "high_ppl_clusters": hi,
                "classification": cls,
            })

        if trace and (step % TRACE_EVERY == 1 or step == 1):
            self._token_trace(step, batch, pre)

        self.perf["steps"] += 1
        self.perf["positions"] += batch["n_positions"]
        self.perf["loss_tokens"] += batch["n_loss_tokens"]
        self.perf["real_tokens"] += batch["n_real_tokens"]
        self.perf["accepted_seqs"] += len(batch["sequences"])
        self.perf["candidates"] += len(batch["opus_decisions"])

        self.log.event("batch_packed", step=step, batch_id=batch["batch_id"],
                       stage=batch["curriculum_stage"],
                       lanes=hash_obj(batch["planned_alloc"])[:8],
                       util=round(batch["n_real_tokens"] / batch["n_positions"], 4),
                       loss_tokens=batch["n_loss_tokens"],
                       loss_before=round(mean_pre, 5), loss_after=round(mean_post, 5))
        accepted = sum(1 for d in batch["opus_decisions"] if d["status"] == "accepted")
        rejected = sum(1 for d in batch["opus_decisions"] if d["status"] == "rejected")
        deferred = sum(1 for d in batch["opus_decisions"]
                       if d["status"].startswith("deferred"))
        overrides = sum(1 for d in batch["opus_decisions"]
                        if d["protected_floor_override"])
        self.log.event("opus_decisions_recorded", step=step, candidates=len(
            batch["opus_decisions"]), accepted=accepted, rejected=rejected,
            deferred=deferred, protected_floor_overrides=overrides)
        return batch

    def _token_trace(self, step: int, batch: dict, pre: List[tuple]) -> None:
        s = batch["sequences"][0]
        losses = pre[0][1]
        tok = self.env["tok"]
        span_of = {}
        for m in s["members"]:
            for t in range(m["dst_start"], m["dst_end"]):
                span_of[t] = m
        n = 0
        for t, lm in enumerate(s["loss_mask"]):
            if not lm:
                continue
            m = span_of.get(t, {})
            loss = float(losses[t])
            self.ledgers["token_trace"].append({
                "type": "token_loss",
                "global_step": step,
                "batch_id": batch["batch_id"],
                "pack_id": s["pack_id"],
                "position": t,
                "position_id": s["position_ids"][t],
                "segment_id": s["segment_ids"][t],
                "token_id": int(s["input_ids"][t]),
                "target_token_id": int(s["input_ids"][t + 1]) if t + 1 < len(
                    s["input_ids"]) else -1,
                "preview": tok.preview(s["input_ids"][t]),
                "role_id": s["role_ids"][t],
                "doc_id": m.get("doc_id", ""),
                "shard_id": m.get("shard_id", ""),
                "lane": s["lane"],
                "loss": round(loss, 8),
                "ppl": round(math.exp(min(20.0, loss)), 6),
                "repeated_pass_number": s["pass_no"],
                "checkpoint_before": self.current_ckpt,
                "curriculum_stage": batch["curriculum_stage"],
            })
            n += 1
        self.log.info("token_trace_written", step=step, tokens=n,
                      pack_id=s["pack_id"])

    # --------------------------------------------------------------- ckpt --
    def checkpoint(self, step: int) -> dict:
        doc = ckpt.save_checkpoint(
            self.branch.branch_id, self.run_id, step, step + 1, self.model,
            self.stream.state(), self.ledgers,
            {"parent_checkpoint": self.current_ckpt,
             "parent_branch": self.branch.parent_branch,
             "fork_step": self.branch.fork_step,
             "mixture_schedule_hash": self.env["mixture"].to_json()["schedule_hash"],
             "tokenizer_hash": self.env["tok"].tokenizer_hash})
        self.ledgers["control"].append({
            "type": "checkpoint_saved", "checkpoint_id": doc["checkpoint_id"],
            "global_step": step, "next_step": step + 1,
            "ledger_offsets": doc["ledger_offsets"],
            "ledger_heads": doc["ledger_heads"],
            "param_hash": doc["param_hash"],
            "checkpoint_hash": doc["checkpoint_hash"]})
        self.current_ckpt = doc["checkpoint_id"]
        self.log.ok("checkpoint_saved", checkpoint_id=doc["checkpoint_id"], step=step,
                    consumption_offset=doc["ledger_offsets"]["consumption"],
                    param_hash=doc["param_hash"][:16])
        return doc

    def validate(self, step: int, val_packs: List[dict]) -> dict:
        """Validation is read-only: forward pass, no gradients, logged access."""
        losses = []
        for p in val_packs:
            l, _, n = self.model.sequence_loss(p["input_ids"], p["position_ids"],
                                               p["segment_ids"], p["loss_mask"])
            if n:
                losses.append(l)
        doc_ids = sorted({m["doc_id"] for p in val_packs for m in p["members"]})
        access = self.env["registry"].record_access(
            f"{self.run_id}:{self.branch.branch_id}", doc_ids,
            "validation_eval", gradient_bearing=False)
        rec = {"type": "validation_eval", "global_step": step,
               "mean_loss": round(float(np.mean(losses)) if losses else 0.0, 8),
               "packs": len(val_packs), "gradient_bearing": False,
               "doc_ids": doc_ids, "access_permitted": access["permitted"]}
        self.ledgers["control"].append(rec)
        self.log.event("validation_eval", step=step, mean_loss=rec["mean_loss"],
                       gradient_bearing=False)
        return rec

    # ---------------------------------------------------------------- perf --
    def flush_perf(self, tag: str) -> dict:
        self.perf["wall_seconds"] = time.perf_counter() - self._t_start
        self.perf["loader_seconds"] = self.stream.loader_seconds
        rec = {"type": "perf_segment", "tag": tag, "branch_id": self.branch.branch_id,
               "run_id": self.run_id, **{k: round(v, 8) if isinstance(v, float) else v
                                         for k, v in self.perf.items()},
               "shard_cache": self.env["store"].stats(),
               "rejections_by_lane": dict(self.stream.rejections_by_lane)}
        self.ledgers["control"].append(rec)
        return rec


def _high_ppl_clusters(seq: dict, losses, tok, k: int = 3) -> List[dict]:
    idx = [t for t, m in enumerate(seq["loss_mask"]) if m]
    if not idx:
        return []
    top = sorted(idx, key=lambda t: -float(losses[t]))[:k]
    out = []
    for t in sorted(top):
        out.append({"position": t, "token_id": int(seq["input_ids"][t]),
                    "preview": tok.preview(seq["input_ids"][t]),
                    "loss": round(float(losses[t]), 6),
                    "ppl": round(math.exp(min(20.0, float(losses[t]))), 4)})
    return out
