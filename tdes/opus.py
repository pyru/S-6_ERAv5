"""OPUS candidate selection with a full audit trail.

OPUS sits *inside* the data path: for each lane quota the loader over-generates
candidates, scores them against a frozen proxy checkpoint, and accepts, defers
or rejects each one.  Rejections are kept - they are the record of what the
selector considered low value, and of what a protected floor had to rescue.

Score = 0.55 * proxy surprise
      + 0.25 * lane deficit against the stage's protected floor / target share
      + 0.20 * novelty (inverse repeated-pass number)
scaled to zero when the lane is not active in the current curriculum stage.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .config import (OPUS_ACCEPT_THRESHOLD, OPUS_DEFER_THRESHOLD, OPUS_PROXY_VERSION)

W_SURPRISE, W_DEFICIT, W_NOVELTY = 0.65, 0.25, 0.10
SURPRISE_REF = 3.0


def surprise_norm(mean_loss: float) -> float:
    return 1.0 - math.exp(-max(0.0, mean_loss) / SURPRISE_REF)


class OpusSelector:
    def __init__(self, proxy_model, proxy_checkpoint_id: str):
        self.proxy = proxy_model
        self.proxy_checkpoint_id = proxy_checkpoint_id
        self.proxy_version = OPUS_PROXY_VERSION
        self._score_cache: Dict[str, float] = {}

    def rebind(self, proxy_model, proxy_checkpoint_id: str) -> None:
        self.proxy = proxy_model
        self.proxy_checkpoint_id = proxy_checkpoint_id
        self._score_cache.clear()

    # ---------------------------------------------------------------- score --
    def proxy_loss(self, pack: dict) -> float:
        key = pack["pack_hash"]
        if key not in self._score_cache:
            loss, _, _ = self.proxy.sequence_loss(
                pack["input_ids"], pack["position_ids"],
                pack["segment_ids"], pack["loss_mask"])
            self._score_cache[key] = float(loss)
        return self._score_cache[key]

    def score_candidate(self, pack: dict, pass_no: int, lane_deficit: float,
                        stage_active: bool) -> dict:
        loss = self.proxy_loss(pack)
        sur = surprise_norm(loss)
        nov = 1.0 / (1.0 + pass_no)
        raw = W_SURPRISE * sur + W_DEFICIT * min(1.0, max(0.0, lane_deficit)) + \
            W_NOVELTY * nov
        score = raw if stage_active else 0.0
        return {"proxy_loss": loss, "surprise": sur, "novelty": nov,
                "lane_deficit": lane_deficit, "stage_active": stage_active,
                "score": score}

    # --------------------------------------------------------------- select --
    def select(self, step: int, stage: str, lane: str, need: int,
               candidates: List[dict], lane_deficit: float,
               floor_protected: bool) -> Tuple[List[dict], List[dict]]:
        """Returns (chosen_packs, decision_records) - all candidates recorded."""
        scored = []
        for cand in candidates:
            pack, pass_no = cand["pack"], cand["pass_no"]
            sc = self.score_candidate(pack, pass_no, lane_deficit, cand["stage_active"])
            status = ("accepted" if sc["score"] >= OPUS_ACCEPT_THRESHOLD
                      else "deferred" if sc["score"] >= OPUS_DEFER_THRESHOLD
                      else "rejected")
            reason = ""
            if status != "accepted":
                if not sc["stage_active"]:
                    reason = "stage_mismatch"
                elif pass_no >= 2:
                    reason = "duplication"
                elif sc["surprise"] < 0.35:
                    reason = "low_proxy_utility"
                else:
                    reason = "below_accept_threshold"
            scored.append({"cand": cand, "sc": sc, "status": status, "reason": reason})

        order = sorted(range(len(scored)),
                       key=lambda i: (-scored[i]["sc"]["score"],
                                      scored[i]["cand"]["pack"]["pack_id"]))
        chosen_idx: List[int] = []
        for want_status in ("accepted", "deferred", "rejected"):
            for i in order:
                if len(chosen_idx) >= need:
                    break
                if i in chosen_idx:
                    continue
                s = scored[i]
                if s["status"] != want_status:
                    continue
                if want_status == "deferred":
                    # a lane that has fallen under its protected floor rescues
                    # candidates OPUS would otherwise have left on the bench
                    if floor_protected:
                        s["final"] = "protected_floor_override"
                        s["override"] = True
                    else:
                        s["final"] = "deferred_promoted"
                        s["reason"] = "quota_fill_from_deferred"
                elif want_status == "rejected":
                    if not floor_protected:
                        continue
                    s["final"] = "protected_floor_override"
                    s["override"] = True
                else:
                    s["final"] = "accepted"
                chosen_idx.append(i)
            if len(chosen_idx) >= need:
                break

        # the loader must still deliver a full batch: backfill is recorded, not hidden
        if len(chosen_idx) < need:
            for i in order:
                if len(chosen_idx) >= need:
                    break
                if i in chosen_idx:
                    continue
                scored[i]["final"] = "quota_backfill"
                chosen_idx.append(i)

        # anything left over that scored well was squeezed out by the quota
        chosen_set = set(chosen_idx)
        for i, s in enumerate(scored):
            if i in chosen_set:
                continue
            if s["status"] == "accepted":
                s["final"] = "rejected"
                s["reason"] = "quota_pressure"
            elif s["status"] == "deferred":
                s["final"] = "deferred"
            else:
                s["final"] = "rejected"

        decisions = []
        chosen_packs = []
        for i, s in enumerate(scored):
            cand, sc = s["cand"], s["sc"]
            pack = cand["pack"]
            rec = {
                "candidate_id": f"cand-{step:05d}-{lane}-{cand['slot']:02d}",
                "global_step": step,
                "curriculum_stage": stage,
                "capability_lane": lane,
                "pack_id": pack["pack_id"],
                "pack_hash": pack["pack_hash"],
                "shard_ids": sorted({m["shard_id"] for m in pack["members"]}),
                "repeated_pass_number": cand["pass_no"],
                "scoring_checkpoint_id": self.proxy_checkpoint_id,
                "proxy_version": self.proxy_version,
                "proxy_loss": round(sc["proxy_loss"], 8),
                "opus_score": round(sc["score"], 8),
                "surprise": round(sc["surprise"], 8),
                "lane_deficit": round(sc["lane_deficit"], 8),
                "novelty": round(sc["novelty"], 8),
                "initial_status": s["status"],
                "status": s.get("final", s["status"]),
                "rejection_reason": s["reason"],
                "protected_floor_override": bool(s.get("override", False)),
                "selected": i in chosen_set,
                "effective_tokens": int(sum(pack["loss_mask"])),
            }
            decisions.append(rec)
            if i in chosen_set:
                chosen_packs.append({"pack": pack, "decision": rec,
                                     "pass_no": cand["pass_no"]})

        chosen_packs.sort(key=lambda c: c["decision"]["candidate_id"])
        return chosen_packs, decisions
