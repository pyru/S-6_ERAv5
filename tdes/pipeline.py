"""Data preparation: corpus -> tokenizer -> shards -> manifests -> mixture -> packs."""
from __future__ import annotations

import json
import os
from typing import List

from .config import (FORK_BRANCH, LANES, MAIN_BRANCH, PATHS, TOTAL_STEPS,
                     VOCAB_SIZE, config_fingerprint)
from .corpus import CANARY, build_corpus, write_corpus
from .hashing import hash_tokens
from .mixture import MixtureSchedule
from .packing import (build_pack_pools, pack_samples, packing_report,
                      verify_pack, write_pack_pools)
from .registry import EvalRegistry
from .shards import (ShardStore, build_shards, run_admission, verify_shard,
                     write_manifest_index)
from .tokenizer import Tokenizer, normalize, tokenize_document, train_tokenizer

ZWJ, ZWNJ = "‍", "‌"
INDIC_PROBES = [
    "ज्ञान क्षेत्र विद्यालय",                    # Devanagari conjuncts
    "क" + ZWJ + "ष",                  # explicit ZWJ
    "क्" + ZWNJ + "ष",           # explicit ZWNJ
    "கணினி மொழி ஆய்வு",                          # Tamil
    "मिश्रित text 123",                          # mixed script + digits
]


def prepare(log) -> dict:
    log.section("PHASE 1 - CORPUS")
    corpus = build_corpus()
    candidates = corpus["train"] + corpus["rejected_seed"]
    path = write_corpus(corpus)
    log.event("corpus_built", train_docs=len(corpus["train"]),
              eval_docs=len(corpus["test"]), validation_docs=len(corpus["validation"]),
              poisoned_candidates=len(corpus["rejected_seed"]), path=os.path.basename(path))

    # ------------------------------------------------------------ tokenizer --
    log.section("PHASE 2 - FROZEN TOKENIZER")
    tok = train_tokenizer([d["text"] for d in corpus["train"]], VOCAB_SIZE)
    tok_path = tok.save()
    frozen_hash = tok.tokenizer_hash
    reloaded = Tokenizer.load()
    same_hash = reloaded.tokenizer_hash == frozen_hash
    roundtrip = all(reloaded.decode(reloaded.encode(p)) == normalize(p)
                    for p in INDIC_PROBES)
    zw_preserved = all(normalize(p).count(ZWJ) == p.count(ZWJ) and
                       normalize(p).count(ZWNJ) == p.count(ZWNJ)
                       for p in INDIC_PROBES) and any(ZWJ in p for p in INDIC_PROBES)
    determinism = all(reloaded.encode(p) == tok.encode(p) for p in INDIC_PROBES)
    log.check(same_hash and determinism, "tokenizer_hash_verified",
              tokenizer_hash=frozen_hash[:24], vocab_size=tok.vocab_size,
              merges=len(tok.merges))
    log.check(roundtrip, "tokenizer_roundtrip_lossless", probes=len(INDIC_PROBES))
    log.check(zw_preserved, "indic_zero_width_preserved")

    # -------------------------------------------------------------- registry --
    log.section("PHASE 3 - EVALUATION FIREWALL REGISTRY")
    reg = EvalRegistry()
    for d in corpus["test"]:
        e = reg.register(d, tok, "never_train")
        log.event("eval_shard_registered", doc_id=d["doc_id"],
                  benchmark_id=e["benchmark_id"], never_train=True,
                  token_hash=e["token_hash"][:16])
    for d in corpus["validation"]:
        reg.register(d, tok, "eval_read_only")
    log.event("validation_registered", docs=len(corpus["validation"]),
              permission="eval_read_only")

    # ------------------------------------------------------------- admission --
    log.section("PHASE 4 - ADMISSION GATE")
    admitted, rejected = run_admission(candidates, reg, log)
    blocked_eval = [r for r in reg.admission_log
                    if not r["admitted"] and any(
                        x.startswith("eval_contamination") or
                        x == "registered_evaluation_document" for x in r["reasons"])]
    log.check(len(blocked_eval) >= 2, "eval_shard_blocked",
              blocked_docs=[b["doc_id"] for b in blocked_eval],
              reasons=sorted({x.split(":")[0] for b in blocked_eval
                              for x in b["reasons"]}))
    log.check(all(d["split"] == "train" for d in admitted), "only_training_split_admitted",
              admitted=len(admitted), rejected=len(rejected))

    # ---------------------------------------------------------------- shards --
    log.section("PHASE 5 - IMMUTABLE TOKENIZED SHARDS")
    manifests = build_shards(admitted, tok, log)
    verified = [verify_shard(m) for m in manifests]
    all_ok = all(v[0] for v in verified)
    log.check(all_ok, "manifests_validated", shards=len(manifests),
              failures=[m["shard_id"] for m, v in zip(manifests, verified) if not v[0]])
    tok_ok = all(m["tokenizer_hash"] == frozen_hash for m in manifests)
    log.check(tok_ok, "shard_tokenizer_hash_uniform", tokenizer_hash=frozen_hash[:24])

    index = write_manifest_index(manifests, tok, {
        "candidates": len(candidates), "admitted": len(admitted),
        "rejected": len(rejected)})
    log.event("manifest_index_written", shards=index["shard_count"],
              total_tokens=index["total_tokens"], merkle_root=index["merkle_root"][:24])

    store = ShardStore(manifests, frozen_hash)

    # post-hoc firewall sweep over every shard that will be trained on
    contaminated = []
    for m in manifests:
        hits = reg.scan_tokens(store.get(m["shard_id"])["tokens"])
        if hits:
            contaminated.append(m["shard_id"])
    log.check(not contaminated, "no_eval_tokens_in_shards",
              scanned_shards=len(manifests), contaminated=contaminated)
    canary_free = all(CANARY not in tok.decode(store.get(m["shard_id"])["tokens"])
                      for m in manifests)
    log.check(canary_free, "canary_absent_from_shards", canary=CANARY[:16])

    # ------------------------------------------------------------- validation --
    val_samples = []
    for d in corpus["validation"]:
        enc = tokenize_document(tok, d)
        val_samples.append({"sample_id": "validation:" + d["doc_id"],
                            "shard_id": "validation-holdout", "doc_id": d["doc_id"],
                            "lane": "validation", "src_start": 0,
                            "src_end": len(enc["tokens"]), "tokens": enc["tokens"],
                            "roles": enc["roles"], "loss": enc["loss"],
                            "structured": False,
                            "token_hash": hash_tokens(enc["tokens"])})
    val_packs = pack_samples(val_samples, "pad_only", "validation")
    vpath = os.path.join(PATHS["packs"], "_validation.jsonl")
    with open(vpath, "w", encoding="utf-8") as fh:
        for p in val_packs:
            p["never_train"] = True
            fh.write(json.dumps(p, separators=(",", ":"), sort_keys=True) + "\n")
    log.event("validation_packs_built", packs=len(val_packs), never_train=True,
              gradient_bearing=False)

    # ---------------------------------------------------------------- mixture --
    log.section("PHASE 6 - MIXTURE TIMELINE COMPILED")
    available = {l: 0 for l in LANES}
    for m in manifests:
        available[m["lane"]] = available.get(m["lane"], 0) + m["token_count"]
    main_mix = MixtureSchedule(MAIN_BRANCH.mixture_override)
    fork_mix = MixtureSchedule(FORK_BRANCH.mixture_override)
    p1 = main_mix.write("main", available, TOTAL_STEPS)
    p2 = fork_mix.write("fork-a", available, TOTAL_STEPS)
    doc = main_mix.to_json(available, TOTAL_STEPS)
    log.event("mixture_compiled", branch="main", stages=len(doc["stages"]),
              schedule_hash=doc["schedule_hash"][:24],
              scarce_lanes=doc["scarcity"]["scarce_lanes"],
              path=os.path.basename(p1))
    log.event("mixture_compiled", branch="fork-a",
              schedule_hash=fork_mix.to_json()["schedule_hash"][:24],
              path=os.path.basename(p2))
    for st in doc["stages"]:
        log.info("curriculum_stage", stage=st["stage"],
                 tokens=f"{st['token_start']}..{st['token_end']}",
                 effective_mixture={k: round(v, 4)
                                    for k, v in st["effective_mixture"].items()},
                 protected_floors=st["protected_floors"],
                 anneal_reserve=st.get("anneal_reserve", {}))
    floors_ok = all(st["effective_mixture"][l] >= st["protected_floors"].get(l, 0) - 1e-9
                    for st in doc["stages"] for l in LANES)
    log.check(floors_ok, "protected_floors_compiled")

    # ------------------------------------------------------------------ packs --
    log.section("PHASE 7 - PACKING")
    pools = build_pack_pools(store, manifests)
    write_pack_pools(pools)
    total_packs = sum(len(v) for v in pools.values())
    violations = {lane: sum(len(verify_pack(p)) for p in packs)
                  for lane, packs in pools.items()}
    log.check(sum(violations.values()) == 0, "packing_invariants_verified",
              packs=total_packs, lanes=len(pools), violations=violations)
    empty = [lane for lane, packs in pools.items() if not packs]
    log.check(not empty, "every_lane_has_a_pack_pool", empty_lanes=empty)
    if empty:
        raise RuntimeError("empty pack pool(s): %s - samples exceed the window" % empty)
    for lane in sorted(pools):
        packs = pools[lane]
        util = sum(p["n_real_tokens"] for p in packs) / sum(p["seq_len"] for p in packs)
        useful = sum(p["n_loss_tokens"] for p in packs) / sum(p["seq_len"] for p in packs)
        log.event("pack_pool_built", lane=lane, policy=packs[0]["policy"],
                  packs=len(packs), utilization=round(util, 4),
                  useful_ratio=round(useful, 4))
    prep = packing_report(store, manifests)
    log.event("packing_report_written", rows=len(prep["rows"]),
              report_hash=prep["report_hash"][:16])

    summary = {
        "tokenizer_hash": frozen_hash,
        "tokenizer_path": os.path.relpath(tok_path, PATHS["art"]).replace("\\", "/"),
        "shard_count": len(manifests),
        "total_tokens": index["total_tokens"],
        "merkle_root": index["merkle_root"],
        "admitted_docs": len(admitted),
        "rejected_docs": len(rejected),
        "pack_pools": {k: len(v) for k, v in pools.items()},
        "validation_packs": len(val_packs),
        "available_tokens_by_lane": available,
        "config_fingerprint": config_fingerprint(),
        "mixture_schedule_hash": doc["schedule_hash"],
        "packing_report_hash": prep["report_hash"],
    }
    reg.write()
    reg.write_admission_report({"blocked_eval_docs": [b["doc_id"] for b in blocked_eval]})
    with open(os.path.join(PATHS["manifests"], "prepare_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    return summary


def load_validation_packs() -> List[dict]:
    path = os.path.join(PATHS["packs"], "_validation.jsonl")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
