"""Packing policies, loss masks, attention masks and position ids.

A *pack* is one fixed-length training window.  It carries not only token ids
but the training meaning of those ids:

* ``segment_ids``  - attention isolation unit (one packed sample).  Two samples
  packed into the same window can never attend to each other.
* ``role_ids``     - which contract the token belongs to (prompt / response /
  tool observation / thought / answer).  Drives the loss mask.
* ``position_ids`` - restart at 0 for every segment, so a packed sample does
  not inherit the positions of whatever preceded it in the window.
* ``loss_mask``    - position ``t`` is loss-bearing iff the *target* ``t+1``
  lives in the same segment and its role is loss-bearing.  This is the
  Session 1 next-token contract made explicit.

Policies: pad_only, concat_chop, greedy, best_fit, structure_preserving,
long_context.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence

from .config import LANE_PACKING_POLICY, LONG_SEQ_LEN, PACKER_VERSION, PATHS, SEQ_LEN
from .hashing import hash_obj, hash_tokens
from .tokenizer import PAD

ROLES = ["<pad>", "text", "user", "assistant", "tool_call", "tool_obs",
         "think", "answer", "eos"]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}

POLICIES = ["pad_only", "concat_chop", "greedy", "best_fit",
            "structure_preserving", "long_context"]


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------
def samples_from_shard(entry: dict) -> List[dict]:
    """Slice a sealed shard into per-document samples."""
    m = entry["manifest"]
    out = []
    for d in m["documents"]:
        s, e = d["token_start"], d["token_end"]
        out.append({
            "sample_id": f"{m['shard_id']}:{d['doc_id']}",
            "shard_id": m["shard_id"],
            "doc_id": d["doc_id"],
            "lane": m["lane"],
            "src_start": s,
            "src_end": e,
            "tokens": entry["tokens"][s:e],
            "roles": entry["roles"][s:e],
            "loss": entry["loss"][s:e],
            "structured": d["structured"],
            "token_hash": d["token_hash"],
        })
    return out


# --------------------------------------------------------------------------
# pack construction
# --------------------------------------------------------------------------
class _Window:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.ids: List[int] = []
        self.roles: List[int] = []
        self.tgt_loss: List[int] = []
        self.seg: List[int] = []
        self.pos: List[int] = []
        self.members: List[dict] = []
        self.boundary_crossings = 0

    @property
    def free(self) -> int:
        return self.seq_len - len(self.ids)

    def add_sample(self, sample: dict, src_off: int = 0, count: int | None = None,
                   new_segment: bool = True) -> None:
        n = len(sample["tokens"]) if count is None else count
        toks = sample["tokens"][src_off:src_off + n]
        roles = sample["roles"][src_off:src_off + n]
        loss = sample["loss"][src_off:src_off + n]
        if not self.seg:
            seg = 1
        elif new_segment:
            seg = self.seg[-1] + 1
        else:
            seg = self.seg[-1]
        dst_start = len(self.ids)
        base_pos = 0 if (new_segment or not self.pos) else self.pos[-1] + 1
        for i, t in enumerate(toks):
            self.ids.append(int(t))
            self.roles.append(ROLE_ID.get(roles[i], ROLE_ID["text"]))
            self.tgt_loss.append(int(loss[i]))
            self.seg.append(seg)
            self.pos.append(base_pos + i)
        if not new_segment and self.members:
            self.boundary_crossings += 1
        self.members.append({
            "sample_id": sample["sample_id"],
            "doc_id": sample["doc_id"],
            "shard_id": sample["shard_id"],
            "src_start": sample["src_start"] + src_off,
            "src_end": sample["src_start"] + src_off + len(toks),
            "dst_start": dst_start,
            "dst_end": dst_start + len(toks),
            "segment_id": seg,
            "token_hash": hash_tokens(toks),
            "partial": len(toks) != len(sample["tokens"]),
        })

    def finalize(self, pack_id: str, policy: str, lane: str,
                 truncated: int = 0, dropped: int = 0) -> dict:
        L = self.seq_len
        n_real = len(self.ids)
        ids = self.ids + [PAD] * (L - n_real)
        roles = self.roles + [ROLE_ID["<pad>"]] * (L - n_real)
        tgt = self.tgt_loss + [0] * (L - n_real)
        seg = self.seg + [0] * (L - n_real)
        pos = self.pos + [0] * (L - n_real)

        loss_mask = [0] * L
        for t in range(L - 1):
            if seg[t] != 0 and seg[t + 1] == seg[t] and tgt[t + 1] == 1:
                loss_mask[t] = 1

        pack = {
            "pack_id": pack_id,
            "policy": policy,
            "lane": lane,
            "seq_len": L,
            "packer_version": PACKER_VERSION,
            "input_ids": ids,
            "loss_mask": loss_mask,
            "segment_ids": seg,
            "position_ids": pos,
            "role_ids": roles,
            "target_loss": tgt,
            "members": self.members,
            "n_real_tokens": n_real,
            "n_pad_tokens": L - n_real,
            "n_loss_tokens": int(sum(loss_mask)),
            "n_segments": len(set(s for s in seg if s)),
            "boundary_crossings": self.boundary_crossings,
            "truncated_samples": truncated,
            "dropped_samples": dropped,
            "utilization": n_real / L,
            "useful_ratio": sum(loss_mask) / L,
            "token_hash": hash_tokens(ids),
        }
        pack["pack_hash"] = hash_obj({
            "input_ids": ids, "loss_mask": loss_mask, "segment_ids": seg,
            "position_ids": pos, "role_ids": roles, "policy": policy,
            "members": [{k: v for k, v in m.items()} for m in self.members],
            "packer_version": PACKER_VERSION,
        })
        return pack


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
def pack_samples(samples: List[dict], policy: str, lane: str,
                 seq_len: int = SEQ_LEN) -> List[dict]:
    if policy == "long_context":
        seq_len = LONG_SEQ_LEN
    fn = {
        "pad_only": _pack_pad_only,
        "concat_chop": _pack_concat_chop,
        "greedy": _pack_greedy,
        "best_fit": _pack_best_fit,
        "structure_preserving": _pack_structure,
        "long_context": _pack_long_context,
    }[policy]
    packs = fn(samples, lane, seq_len)
    for i, p in enumerate(packs):
        p["pack_index"] = i
    return packs


def _pid(lane: str, policy: str, i: int) -> str:
    return f"pack-{lane}-{policy}-{i:04d}"


def _pack_pad_only(samples, lane, seq_len):
    packs, trunc = [], 0
    for i, s in enumerate(samples):
        w = _Window(seq_len)
        n = min(len(s["tokens"]), seq_len)
        if n < len(s["tokens"]):
            trunc += 1
        w.add_sample(s, 0, n, new_segment=True)
        packs.append(w.finalize(_pid(lane, "pad_only", i), "pad_only", lane,
                                truncated=int(n < len(s["tokens"]))))
    return packs


def _pack_concat_chop(samples, lane, seq_len):
    """Join documents (each already EOS-terminated) and chop fixed windows."""
    packs = []
    w = _Window(seq_len)
    idx = 0
    first_in_window = True
    for s in samples:
        off = 0
        while off < len(s["tokens"]):
            take = min(w.free, len(s["tokens"]) - off)
            w.add_sample(s, off, take, new_segment=first_in_window)
            first_in_window = False
            off += take
            if w.free == 0:
                packs.append(w.finalize(_pid(lane, "concat_chop", idx),
                                        "concat_chop", lane))
                idx += 1
                w = _Window(seq_len)
                first_in_window = True
    if w.ids:
        packs.append(w.finalize(_pid(lane, "concat_chop", idx), "concat_chop", lane))
    return packs


def _pack_greedy(samples, lane, seq_len):
    """First-fit: order-dependent, never splits a sample."""
    windows: List[_Window] = []
    dropped = 0
    for s in samples:
        n = len(s["tokens"])
        if n > seq_len:
            dropped += 1
            continue
        for w in windows:
            if w.free >= n:
                w.add_sample(s)
                break
        else:
            w = _Window(seq_len)
            w.add_sample(s)
            windows.append(w)
    return [w.finalize(_pid(lane, "greedy", i), "greedy", lane,
                       dropped=(dropped if i == 0 else 0))
            for i, w in enumerate(windows)]


def _pack_best_fit(samples, lane, seq_len):
    """Best-fit-decreasing: tightest remaining space wins."""
    windows: List[_Window] = []
    dropped = 0
    order = sorted(samples, key=lambda s: (-len(s["tokens"]), s["sample_id"]))
    for s in order:
        n = len(s["tokens"])
        if n > seq_len:
            dropped += 1
            continue
        best, best_free = None, None
        for w in windows:
            if w.free >= n and (best_free is None or w.free < best_free):
                best, best_free = w, w.free
        if best is None:
            best = _Window(seq_len)
            windows.append(best)
        best.add_sample(s)
    return [w.finalize(_pid(lane, "best_fit", i), "best_fit", lane,
                       dropped=(dropped if i == 0 else 0))
            for i, w in enumerate(windows)]


def _pack_structure(samples, lane, seq_len):
    """Best-fit, but structured samples are never split and never truncated."""
    windows: List[_Window] = []
    dropped = 0
    order = sorted(samples, key=lambda s: (-len(s["tokens"]), s["sample_id"]))
    for s in order:
        n = len(s["tokens"])
        if n > seq_len:
            dropped += 1
            continue
        best, best_free = None, None
        for w in windows:
            if w.free >= n and (best_free is None or w.free < best_free):
                best, best_free = w, w.free
        if best is None:
            best = _Window(seq_len)
            windows.append(best)
        best.add_sample(s, new_segment=True)
    return [w.finalize(_pid(lane, "structure_preserving", i),
                       "structure_preserving", lane,
                       dropped=(dropped if i == 0 else 0))
            for i, w in enumerate(windows)]


def _pack_long_context(samples, lane, seq_len):
    """Only long samples; every unused slot in a long window is expensive."""
    longs = [s for s in samples if len(s["tokens"]) >= seq_len // 4]
    if not longs:
        longs = samples
    return [p for p in _pack_best_fit(longs, lane, seq_len)]


# --------------------------------------------------------------------------
# masks + verification
# --------------------------------------------------------------------------
def build_attention_mask(segment_ids: Sequence[int]) -> List[List[bool]]:
    """Causal, block-diagonal by segment.  Pad attends to nothing."""
    L = len(segment_ids)
    return [[bool(j <= i and segment_ids[i] != 0 and segment_ids[i] == segment_ids[j])
             for j in range(L)] for i in range(L)]


def verify_pack(pack: dict) -> List[str]:
    """Return a list of invariant violations (empty means correct)."""
    v: List[str] = []
    L = pack["seq_len"]
    ids, lm = pack["input_ids"], pack["loss_mask"]
    seg, pos, roles = pack["segment_ids"], pack["position_ids"], pack["role_ids"]
    tgt = pack["target_loss"]
    if not (len(ids) == len(lm) == len(seg) == len(pos) == len(roles) == L):
        v.append("length_mismatch")
        return v
    if any(m not in (0, 1) for m in lm):
        v.append("loss_mask_not_binary")

    # padding
    for t in range(L):
        if seg[t] == 0:
            if ids[t] != PAD:
                v.append(f"pad_segment_non_pad_token@{t}")
            if lm[t] != 0:
                v.append(f"loss_on_pad@{t}")
            if roles[t] != ROLE_ID["<pad>"]:
                v.append(f"pad_role_mismatch@{t}")
    if pack["n_real_tokens"] + pack["n_pad_tokens"] != L:
        v.append("token_conservation")
    if any(ids[t] == PAD and seg[t] != 0 for t in range(L)):
        v.append("pad_token_inside_segment")
    # padding must be a suffix
    real = [t for t in range(L) if seg[t] != 0]
    if real and real != list(range(len(real))):
        v.append("padding_not_suffix")

    # position ids restart per segment and increase by one
    for t in range(L):
        if seg[t] == 0:
            continue
        if t == 0 or seg[t - 1] != seg[t]:
            if pos[t] != 0:
                v.append(f"position_not_reset@{t}")
        elif pos[t] != pos[t - 1] + 1:
            v.append(f"position_not_contiguous@{t}")

    # loss mask == next-token contract
    for t in range(L):
        want = 1 if (t + 1 < L and seg[t] != 0 and seg[t + 1] == seg[t]
                     and tgt[t + 1] == 1) else 0
        if lm[t] != want:
            v.append(f"loss_mask_contract@{t}")
    if L and lm[L - 1] != 0:
        v.append("loss_on_last_position")

    # segment boundaries: last position of each segment is never loss-bearing
    for t in range(L - 1):
        if seg[t] != 0 and seg[t + 1] != seg[t] and lm[t] != 0:
            v.append(f"loss_across_segment_boundary@{t}")

    # attention isolation
    att = build_attention_mask(seg)
    for i in range(L):
        for j in range(L):
            if att[i][j] and (j > i or seg[i] != seg[j] or seg[i] == 0):
                v.append(f"attention_leak@{i},{j}")
                break

    # member spans reconstruct the window exactly
    covered = 0
    for m in pack["members"]:
        span = ids[m["dst_start"]:m["dst_end"]]
        if hash_tokens(span) != m["token_hash"]:
            v.append("member_span_hash_mismatch:" + m["sample_id"])
        covered += m["dst_end"] - m["dst_start"]
    if covered != pack["n_real_tokens"]:
        v.append("member_coverage_mismatch")

    if hash_tokens(ids) != pack["token_hash"]:
        v.append("token_hash_mismatch")
    return v


# --------------------------------------------------------------------------
# pools + reports
# --------------------------------------------------------------------------
def build_pack_pools(store, manifests: List[dict],
                     lane_policy: Dict[str, str] | None = None) -> Dict[str, List[dict]]:
    """Immutable, ordered pack pool per lane, using each lane's policy."""
    lane_policy = lane_policy or LANE_PACKING_POLICY
    by_lane: Dict[str, List[dict]] = {}
    for m in sorted(manifests, key=lambda x: x["shard_id"]):
        entry = store.get(m["shard_id"])
        by_lane.setdefault(m["lane"], []).extend(samples_from_shard(entry))
    pools = {}
    for lane in sorted(by_lane):
        pools[lane] = pack_samples(by_lane[lane], lane_policy[lane], lane)
    return pools


def write_pack_pools(pools: Dict[str, List[dict]]) -> Dict[str, str]:
    out = {}
    for lane, packs in pools.items():
        path = os.path.join(PATHS["packs"], f"{lane}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for p in packs:
                fh.write(json.dumps(p, separators=(",", ":"), sort_keys=True) + "\n")
        out[lane] = path
    return out


def load_pack_pools(lanes: Iterable[str]) -> Dict[str, List[dict]]:
    pools = {}
    for lane in lanes:
        path = os.path.join(PATHS["packs"], f"{lane}.jsonl")
        with open(path, "r", encoding="utf-8") as fh:
            pools[lane] = [json.loads(l) for l in fh if l.strip()]
    return pools


def packing_report(store, manifests: List[dict]) -> dict:
    """Compare every policy on every data type - the packing lab, as data."""
    by_lane: Dict[str, List[dict]] = {}
    for m in sorted(manifests, key=lambda x: x["shard_id"]):
        entry = store.get(m["shard_id"])
        by_lane.setdefault(m["lane"], []).extend(samples_from_shard(entry))

    rows = []
    for lane in sorted(by_lane):
        samples = by_lane[lane]
        structured = any(s["structured"] for s in samples)
        for policy in POLICIES:
            packs = pack_samples(samples, policy, lane)
            if not packs:
                continue
            tot = sum(p["seq_len"] for p in packs)
            real = sum(p["n_real_tokens"] for p in packs)
            lossb = sum(p["n_loss_tokens"] for p in packs)
            violations = sum(len(verify_pack(p)) for p in packs)
            splits = sum(1 for p in packs for m in p["members"] if m["partial"])
            rows.append({
                "lane": lane,
                "policy": policy,
                "data_type": "structured" if structured else "plain_text",
                "packs": len(packs),
                "window": packs[0]["seq_len"],
                "total_positions": tot,
                "real_tokens": real,
                "pad_tokens": tot - real,
                "loss_bearing_tokens": lossb,
                "utilization": round(real / tot, 6),
                "useful_ratio": round(lossb / tot, 6),
                "wasted_positions": tot - real,
                "boundary_crossings": sum(p["boundary_crossings"] for p in packs),
                "split_samples": splits,
                "dropped_samples": sum(p["dropped_samples"] for p in packs),
                "invariant_violations": violations,
                "structure_safe": (not structured) or (policy in
                                                       ("pad_only", "structure_preserving",
                                                        "long_context")),
                "selected_for_training": policy == LANE_PACKING_POLICY.get(lane),
            })
    doc = {"schema": "tdes-packing-report/1", "packer_version": PACKER_VERSION,
           "rows": rows}
    doc["report_hash"] = hash_obj(rows)
    path = os.path.join(PATHS["manifests"], "packing_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return doc
