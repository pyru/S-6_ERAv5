"""Mixture timeline compiler (Session 5 -> executable per-step quotas).

Human-language curriculum ("reasoning-heavy midtrain") is compiled into an
integer number of sequences per lane per optimizer step, with:

* protected floors enforced at compile time (weights) *and* at serve time
  (OPUS protected-floor override),
* linear warmup blending across stage transitions,
* annealing reserves that withhold part of each scarce lane's pack pool until
  the anneal stage,
* a scarcity report that flags lanes whose demand exceeds available tokens.

``alloc(step)`` is a pure function of the step index, which is what makes the
stream replayable without replaying the trainer.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .config import LANES, PATHS, SEQS_PER_STEP, SEQ_LEN, TOKENS_PER_STEP, TOTAL_STEPS
from .hashing import hash_obj

_TPS = TOKENS_PER_STEP     # stage boundaries are expressed in whole steps

STAGES: List[dict] = [
    {
        "stage": "foundation",
        "index": 0,
        "token_start": 0,
        "token_end": 16 * _TPS,
        "sequence_length": SEQ_LEN,
        "mixture": {"general_web": 0.42, "code": 0.20, "math_science": 0.14,
                    "indic": 0.12, "agentic": 0.04, "reasoning": 0.08},
        "protected_floors": {"indic": 0.10, "agentic": 0.03, "reasoning": 0.05},
        "warmup_tokens": 0,
        "anneal_reserve": {},
    },
    {
        "stage": "reasoning-heavy-midtrain",
        "index": 1,
        "token_start": 16 * _TPS,
        "token_end": 36 * _TPS,
        "sequence_length": SEQ_LEN,
        "mixture": {"general_web": 0.28, "code": 0.22, "math_science": 0.18,
                    "indic": 0.12, "agentic": 0.06, "reasoning": 0.14},
        "protected_floors": {"indic": 0.12, "agentic": 0.05, "reasoning": 0.12},
        "warmup_tokens": 4 * _TPS,
        "anneal_reserve": {"reasoning": 0.25, "agentic": 0.25},
    },
    {
        "stage": "anneal",
        "index": 2,
        "token_start": 36 * _TPS,
        "token_end": (TOTAL_STEPS + 8) * _TPS,
        "sequence_length": SEQ_LEN,
        "mixture": {"general_web": 0.18, "code": 0.20, "math_science": 0.18,
                    "indic": 0.16, "agentic": 0.10, "reasoning": 0.18},
        "protected_floors": {"indic": 0.15, "agentic": 0.08, "reasoning": 0.16},
        "warmup_tokens": 2 * _TPS,
        "anneal_reserve": {},
        "release_reserves": True,
    },
]


def enforce_floors(weights: Dict[str, float], floors: Dict[str, float]) -> Dict[str, float]:
    """Raise floored lanes to their floor; take the deficit from the rest."""
    w = {l: float(weights.get(l, 0.0)) for l in LANES}
    total = sum(w.values())
    w = {l: v / total for l, v in w.items()}
    raised = {l: max(w[l], floors.get(l, 0.0)) for l in LANES}
    excess = sum(raised.values()) - 1.0
    if excess > 1e-12:
        free = {l: w[l] for l in LANES if l not in floors}
        pool = free if sum(free.values()) > 0 else {l: w[l] for l in LANES}
        s = sum(pool.values())
        for l in pool:
            raised[l] = max(0.0, raised[l] - excess * (pool[l] / s))
    tot = sum(raised.values())
    return {l: raised[l] / tot for l in LANES}


class MixtureSchedule:
    def __init__(self, override: Dict[str, float] | None = None):
        self.override = dict(override or {})
        self.stages = []
        for st in STAGES:
            mix = dict(st["mixture"])
            if self.override:
                mix.update(self.override)
                s = sum(mix.values())
                mix = {k: v / s for k, v in mix.items()}
            eff = enforce_floors(mix, st["protected_floors"])
            rec = dict(st)
            rec["requested_mixture"] = mix
            rec["effective_mixture"] = eff
            self.stages.append(rec)
        self._cum_cache: Dict[int, Dict[str, int]] = {0: {l: 0 for l in LANES}}
        self._ideal_cache: Dict[int, Dict[str, float]] = {0: {l: 0.0 for l in LANES}}

    # ------------------------------------------------------------- staging --
    def stage_for_tokens(self, tokens: int) -> dict:
        for st in self.stages:
            if st["token_start"] <= tokens < st["token_end"]:
                return st
        return self.stages[-1]

    def stage_for_step(self, step: int) -> dict:
        return self.stage_for_tokens((step - 1) * TOKENS_PER_STEP)

    def weights_at_step(self, step: int) -> Dict[str, float]:
        tokens = (step - 1) * TOKENS_PER_STEP
        st = self.stage_for_tokens(tokens)
        idx = st["index"]
        if idx == 0 or st["warmup_tokens"] <= 0:
            return dict(st["effective_mixture"])
        into = tokens - st["token_start"]
        if into >= st["warmup_tokens"]:
            return dict(st["effective_mixture"])
        alpha = into / float(st["warmup_tokens"])
        prev = self.stages[idx - 1]["effective_mixture"]
        cur = st["effective_mixture"]
        blend = {l: (1.0 - alpha) * prev[l] + alpha * cur[l] for l in LANES}
        s = sum(blend.values())
        return {l: v / s for l, v in blend.items()}

    # ---------------------------------------------------------- allocation --
    def _cum(self, step: int) -> Dict[str, int]:
        """Cumulative sequence allocation after `step` steps (pure in `step`)."""
        if step in self._cum_cache:
            return self._cum_cache[step]
        prev = self._cum(step - 1)
        prev_ideal = self._ideal_cache[step - 1]
        w = self.weights_at_step(step)
        ideal = {l: prev_ideal[l] + w[l] * SEQS_PER_STEP for l in LANES}
        # greedy largest-deficit assignment: exact per-step sum, monotone in step
        cur = dict(prev)
        for _ in range(SEQS_PER_STEP):
            lane = max(LANES, key=lambda l: (ideal[l] - cur[l], l))
            cur[lane] += 1
        self._ideal_cache[step] = ideal
        self._cum_cache[step] = cur
        return cur

    def alloc(self, step: int) -> Dict[str, int]:
        cur, prev = self._cum(step), self._cum(step - 1)
        return {l: cur[l] - prev[l] for l in LANES}

    def cum_alloc(self, step: int) -> Dict[str, int]:
        return dict(self._cum(step))

    # ------------------------------------------------------------ reserves --
    def reserve_fraction(self, lane: str, step: int) -> float:
        st = self.stage_for_step(step)
        if st.get("release_reserves"):
            return 0.0
        # a lane reserved by *any* future stage is withheld until that stage
        frac = 0.0
        for s in self.stages:
            if s["index"] >= st["index"]:
                frac = max(frac, float(s.get("anneal_reserve", {}).get(lane, 0.0)))
        return frac

    # ------------------------------------------------------------- reports --
    def scarcity_report(self, available_tokens: Dict[str, int],
                        total_steps: int = TOTAL_STEPS) -> dict:
        demand = {l: 0 for l in LANES}
        for s in range(1, total_steps + 1):
            for l, n in self.alloc(s).items():
                demand[l] += n * SEQ_LEN
        rows = []
        for l in LANES:
            avail = int(available_tokens.get(l, 0))
            d = demand[l]
            rows.append({
                "lane": l,
                "demanded_tokens": d,
                "available_tokens": avail,
                "repetition_factor": round(d / avail, 4) if avail else None,
                "scarce": bool(avail and d > avail),
                "resolution": ("repeat_existing_data" if avail and d > avail
                               else "single_pass_sufficient"),
            })
        return {"rows": rows,
                "scarce_lanes": [r["lane"] for r in rows if r["scarce"]]}

    def to_json(self, available_tokens: Dict[str, int] | None = None,
                total_steps: int = TOTAL_STEPS) -> dict:
        per_step = []
        for s in range(1, total_steps + 1):
            st = self.stage_for_step(s)
            per_step.append({
                "step": s,
                "stage": st["stage"],
                "token_start": (s - 1) * TOKENS_PER_STEP,
                "weights": {k: round(v, 6) for k, v in self.weights_at_step(s).items()},
                "alloc_sequences": self.alloc(s),
                "reserve_fraction": {l: self.reserve_fraction(l, s) for l in LANES},
            })
        doc = {
            "schema": "tdes-mixture-schedule/1",
            "seqs_per_step": SEQS_PER_STEP,
            "tokens_per_step": TOKENS_PER_STEP,
            "seq_len": SEQ_LEN,
            "override": self.override,
            "stages": [{k: v for k, v in st.items()} for st in self.stages],
            "per_step": per_step,
            "cumulative_alloc": self.cum_alloc(total_steps),
        }
        if available_tokens is not None:
            doc["scarcity"] = self.scarcity_report(available_tokens, total_steps)
        doc["schedule_hash"] = hash_obj({k: v for k, v in doc.items()
                                         if k != "schedule_hash"})
        return doc

    def write(self, name: str, available_tokens: Dict[str, int] | None = None,
              total_steps: int = TOTAL_STEPS) -> str:
        doc = self.to_json(available_tokens, total_steps)
        path = os.path.join(PATHS["manifests"], f"mixture_schedule_{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        return path
