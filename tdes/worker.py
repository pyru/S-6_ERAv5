"""Process-level entry points.

Each phase runs as its own OS process so the "crash" is a real, abrupt process
death (``os._exit``) with no cleanup, and recovery has to come from durable
state on disk: checkpoints + hash-chained ledgers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from . import checkpoint as ckpt
from .config import (CHECKPOINT_EVERY, FORK_BRANCH, MAIN_BRANCH, PATHS,
                     TOTAL_STEPS, ensure_dirs)
from .dataloader import batch_fingerprint
from .hashing import hash_obj
from .pipeline import load_validation_packs, prepare
from .runlog import RunLog
from .trainer import Trainer, build_env, new_model, proxy_step_for
from .ledger import LedgerSet
from .dataloader import TrainingStream
from .opus import OpusSelector

CRASH_EXIT_CODE = 70


def _report(name: str, doc: dict) -> str:
    path = os.path.join(PATHS["reports"], name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return path


# --------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    log = RunLog(append=True)
    prepare(log)
    log.close()
    return 0


# --------------------------------------------------------------------------
def cmd_train(args) -> int:
    log = RunLog(append=True)
    log.section("PHASE 8 - TRAINING RUN (branch=%s)" % MAIN_BRANCH.branch_id)
    tr = Trainer(MAIN_BRANCH, args.run_id, log)
    tr.bootstrap()
    val = load_validation_packs()
    log.info("run_started", run_id=args.run_id, branch=MAIN_BRANCH.branch_id,
             until=args.until, crash_at=args.crash_at)

    for step in range(1, args.until + 1):
        tr.consume(step)
        if step % CHECKPOINT_EVERY == 0:
            tr.validate(step, val)
            tr.checkpoint(step)
        if args.crash_at and step == args.crash_at:
            tr.flush_perf("pre_crash")
            last = ckpt.list_checkpoints(MAIN_BRANCH.branch_id)
            committed = [c for c in last if c <= step]
            log.event("crash_simulated", step=step,
                      last_checkpoint=ckpt.ckpt_id(MAIN_BRANCH.branch_id,
                                                   max(committed)),
                      uncheckpointed_steps=step - max(committed),
                      exit_code=CRASH_EXIT_CODE)
            log.close()
            sys.stdout.flush()
            os._exit(CRASH_EXIT_CODE)   # abrupt: no finally blocks, no flushing

    tr.flush_perf("train")
    log.close()
    return 0


# --------------------------------------------------------------------------
def cmd_resume(args) -> int:
    log = RunLog(append=True)
    log.section("PHASE 9 - CRASH RECOVERY / RESUME")
    branch = MAIN_BRANCH
    tr = Trainer(branch, args.run_id, log)

    doc = ckpt.latest_checkpoint(branch.branch_id)
    if doc is None:
        log.fail("resume_no_checkpoint")
        return 1
    tr.restore(doc["global_step"])
    next_step = doc["next_step"]
    log.event("run_resumed", checkpoint_id=doc["checkpoint_id"],
              restored_step=doc["global_step"], next_step=next_step,
              param_hash=doc["param_hash"][:16],
              consumption_offset=doc["ledger_offsets"]["consumption"])

    # 1. the chain up to the committed offset must still verify
    chain_ok = True
    for name, off in doc["ledger_offsets"].items():
        res = tr.ledgers[name].verify_chain(off)
        head_ok = res["head"] == doc["ledger_heads"][name]
        chain_ok = chain_ok and res["ok"] and head_ok
        if not (res["ok"] and head_ok):
            log.fail("ledger_chain_verified", ledger=name, reason=res["reason"])
    log.check(chain_ok, "ledger_chain_verified_at_checkpoint",
              offsets=doc["ledger_offsets"])

    # 2. what the crashed process had already written past the checkpoint
    physical = tr.ledgers["consumption"].read()
    orphan = physical[doc["ledger_offsets"]["consumption"]:]
    orphan_steps = sorted({r["payload"]["global_step"] for r in orphan})
    expected = None
    for r in orphan:
        if r["payload"].get("type") == "batch_served" and \
                r["payload"]["global_step"] == next_step:
            expected = r["payload"]
    log.event("uncommitted_records_found", records=len(orphan),
              steps=orphan_steps,
              expected_next_batch_id=expected["batch_id"] if expected else None)

    # 3. supersede them - the ledger stays append-only, the stream does not repeat
    rb = tr.ledgers.supersede_after(doc["ledger_offsets"], "crash_recovery")
    log.event("ledger_rollback_recorded",
              superseded=rb["superseded"], reason="crash_recovery")

    # 4. serve the next batch and prove it is exactly the expected batch.
    #    the probe must not advance the stream, so its state is snapshotted.
    snapshot = tr.stream.state()
    batch = tr.serve(next_step)
    fp = batch_fingerprint(batch)
    tr.stream.load_state(snapshot)
    matched = bool(expected) and (
        fp["batch_id"] == expected["batch_id"] and
        fp["batch_hash"] == expected["batch_hash"] and
        fp["token_hash"] == expected["token_hash"])
    log.check(matched, "resume_next_batch_matched",
              expected_batch_id=expected["batch_id"] if expected else None,
              resumed_batch_id=fp["batch_id"],
              expected_token_hash=(expected["token_hash"][:16] if expected else None),
              resumed_token_hash=fp["token_hash"][:16])
    resume_report = {
        "branch": branch.branch_id,
        "restored_checkpoint": doc["checkpoint_id"],
        "restored_step": doc["global_step"],
        "next_step": next_step,
        "expected_batch": {k: expected[k] for k in
                           ("batch_id", "batch_hash", "token_hash", "global_step")}
        if expected else None,
        "resumed_batch": {"batch_id": fp["batch_id"], "batch_hash": fp["batch_hash"],
                          "token_hash": fp["token_hash"],
                          "global_step": fp["global_step"]},
        "matched": matched,
        "superseded_records": rb["superseded"],
        "orphan_steps": orphan_steps,
        "ledger_chain_ok": chain_ok,
    }

    # 5. finish the run; consume(next_step) re-serves the very same batch
    val = load_validation_packs()
    for step in range(next_step, args.until + 1):
        tr.consume(step)
        if step % CHECKPOINT_EVERY == 0:
            tr.validate(step, val)
            tr.checkpoint(step)
    tr.flush_perf("resume")

    # 6. no skipped and no repeated batches in the effective stream
    eff = tr.ledgers.effective_consumption()
    served = [r["payload"] for r in eff if r["payload"]["type"] == "batch_served"]
    steps = [s["global_step"] for s in served]
    contiguous = steps == list(range(1, args.until + 1))
    dupes = sorted({s for s in steps if steps.count(s) > 1})
    log.check(contiguous and not dupes, "no_skipped_or_repeated_batches",
              effective_steps=len(steps), expected=args.until,
              duplicates=dupes,
              missing=[s for s in range(1, args.until + 1) if s not in steps])
    resume_report.update({"effective_steps": len(steps), "contiguous": contiguous,
                          "duplicate_steps": dupes,
                          "final_step": args.until})
    _report("resume_report.json", resume_report)
    log.close()
    return 0 if matched and contiguous else 1


# --------------------------------------------------------------------------
def _rebuild_stream(branch):
    """An independent reconstruction of the stream - no trainer, no gradients."""
    env = build_env(branch)
    opus = OpusSelector(new_model(), ckpt.ckpt_id(branch.branch_id, 0))
    stream = TrainingStream(branch, env["mixture"], env["pools"], env["registry"],
                            opus, env["tok"].tokenizer_hash)
    return env, opus, stream


def cmd_replay(args) -> int:
    log = RunLog(append=True)
    log.section("PHASE 10 - REPLAY OF HISTORICAL DATA STREAM")
    branch = MAIN_BRANCH
    env, opus, stream = _rebuild_stream(branch)
    ledgers = LedgerSet(branch.branch_id)

    original: Dict[int, dict] = {}
    for r in ledgers.effective_consumption():
        p = r["payload"]
        if p["type"] == "batch_served":
            original[p["global_step"]] = p
    micro: Dict[int, List[dict]] = {}
    for r in ledgers.effective_consumption():
        p = r["payload"]
        if p["type"] == "microbatch_consumed":
            micro.setdefault(p["global_step"], []).append(p)

    bound = -1
    rows = []
    for step in range(1, args.to + 1):
        ps = proxy_step_for(step)
        if ps != bound:
            doc = ckpt.load_checkpoint(branch.branch_id, ps)
            m = new_model()
            ckpt.restore_model(m, doc)
            opus.rebind(m, doc["checkpoint_id"])
            bound = ps
        batch = stream.build_batch(step, ckpt.ckpt_id(branch.branch_id, ps),
                                   args.run_id + "-replay")
        if step < args.frm:
            continue
        fp = batch_fingerprint(batch)
        orig = original.get(step, {})
        omb = sorted(micro.get(step, []), key=lambda x: x["microbatch_id"])
        rmb = sorted([r for r in _micro_payloads(batch)],
                     key=lambda x: x["microbatch_id"])
        spans_match = ([(m["shard_id"], m["doc_id"], m["src_start"], m["src_end"])
                        for mb in omb for m in mb["token_spans"]] ==
                       [(m["shard_id"], m["doc_id"], m["src_start"], m["src_end"])
                        for mb in rmb for m in mb["token_spans"]])
        row = {
            "step": step,
            "original_batch_id": orig.get("batch_id"),
            "replay_batch_id": fp["batch_id"],
            "batch_id_match": orig.get("batch_id") == fp["batch_id"],
            "original_batch_hash": orig.get("batch_hash"),
            "replay_batch_hash": fp["batch_hash"],
            "batch_hash_match": orig.get("batch_hash") == fp["batch_hash"],
            "original_token_hash": orig.get("token_hash"),
            "replay_token_hash": fp["token_hash"],
            "token_hash_match": orig.get("token_hash") == fp["token_hash"],
            "microbatch_ids_match": [m["microbatch_id"] for m in omb] ==
                                    [m["microbatch_id"] for m in rmb],
            "loss_mask_hash_match": [m["loss_mask_hash"] for m in omb] ==
                                    [m["loss_mask_hash"] for m in rmb],
            "token_spans_match": spans_match,
            "n_token_spans": sum(len(mb["token_spans"]) for mb in rmb),
        }
        row["all_match"] = all(row[k] for k in
                               ("batch_id_match", "batch_hash_match",
                                "token_hash_match", "microbatch_ids_match",
                                "loss_mask_hash_match", "token_spans_match"))
        rows.append(row)
        log.event("replay_step", step=step, batch_id=fp["batch_id"],
                  match=row["all_match"])

    ok = bool(rows) and all(r["all_match"] for r in rows)
    log.check(ok, "replay_hash_matched", interval=f"{args.frm}..{args.to}",
              steps=len(rows),
              mismatches=[r["step"] for r in rows if not r["all_match"]])
    doc = {"branch": branch.branch_id, "interval": [args.frm, args.to],
           "reconstruction": "independent (no trainer, no gradients)",
           "rows": rows, "all_match": ok,
           "replay_digest": hash_obj([r["replay_batch_hash"] for r in rows]),
           "original_digest": hash_obj([r["original_batch_hash"] for r in rows])}
    _report("replay_report.json", doc)
    ledgers["control"].append({"type": "replay_completed", "interval": [args.frm, args.to],
                               "all_match": ok,
                               "replay_digest": doc["replay_digest"]})
    log.close()
    return 0 if ok else 1


def _micro_payloads(batch: dict) -> List[dict]:
    from .dataloader import consumption_records
    return [r for r in consumption_records(batch)
            if r["type"] == "microbatch_consumed"]


# --------------------------------------------------------------------------
def cmd_fork(args) -> int:
    log = RunLog(append=True)
    log.section("PHASE 11 - FORK FROM AN EARLIER CHECKPOINT")
    parent = MAIN_BRANCH
    fork = FORK_BRANCH
    tr = Trainer(fork, args.run_id + "-fork", log)
    src = tr.restore_from(parent.branch_id, fork.fork_step)
    tr.current_ckpt = src["checkpoint_id"]
    tr.stream.load_state(src["stream_state"])
    log.event("branch_forked", parent_branch=parent.branch_id,
              parent_checkpoint=src["checkpoint_id"], fork_step=fork.fork_step,
              new_branch=fork.branch_id, seed=fork.seed,
              mixture_override=fork.mixture_override,
              inherited_param_hash=src["param_hash"][:16])
    tr.ledgers["control"].append({
        "type": "branch_forked", "parent_branch": parent.branch_id,
        "parent_checkpoint": src["checkpoint_id"], "fork_step": fork.fork_step,
        "branch_id": fork.branch_id, "seed": fork.seed,
        "mixture_override": fork.mixture_override,
        "inherited_param_hash": src["param_hash"]})

    parent_ledgers = LedgerSet(parent.branch_id)
    parent_served = {p["payload"]["global_step"]: p["payload"]
                     for p in parent_ledgers.effective_consumption()
                     if p["payload"]["type"] == "batch_served"}

    val = load_validation_packs()
    rows = []
    for step in range(fork.fork_step + 1, fork.fork_step + args.steps + 1):
        batch = tr.consume(step)
        po = parent_served.get(step, {})
        rows.append({"step": step, "fork_batch_id": batch["batch_id"],
                     "main_batch_id": po.get("batch_id"),
                     "diverged": batch["batch_id"] != po.get("batch_id"),
                     "fork_lanes": batch["planned_alloc"],
                     "main_lanes": po.get("planned_alloc")})
        if step % CHECKPOINT_EVERY == 0:
            tr.validate(step, val)
            tr.checkpoint(step)
    tr.flush_perf("fork")

    diverged = all(r["diverged"] for r in rows)
    log.check(diverged, "fork_stream_diverged", steps=len(rows),
              branch=fork.branch_id, parent=parent.branch_id)
    log.check(True, "fork_model_state_inherited",
              parent_checkpoint=src["checkpoint_id"],
              param_hash=src["param_hash"][:16])
    _report("fork_report.json", {
        "parent_branch": parent.branch_id, "fork_branch": fork.branch_id,
        "fork_step": fork.fork_step, "parent_checkpoint": src["checkpoint_id"],
        "inherited_param_hash": src["param_hash"],
        "mixture_override": fork.mixture_override,
        "rows": rows, "all_diverged": diverged})
    log.close()
    return 0 if diverged else 1


# --------------------------------------------------------------------------
def cmd_audit(args) -> int:
    from .audit import run_audit
    log = RunLog(append=True)
    rc = run_audit(log)
    log.close()
    return rc


def cmd_perf(args) -> int:
    from .perf import build_performance
    log = RunLog(append=True)
    rc = build_performance(log)
    log.close()
    return rc


def cmd_evidence(args) -> int:
    from .evidence import build_evidence
    log = RunLog(append=True)
    rc = build_evidence(log)
    log.close()
    return rc


COMMANDS = {"prepare": cmd_prepare, "train": cmd_train, "resume": cmd_resume,
            "replay": cmd_replay, "fork": cmd_fork, "audit": cmd_audit,
            "perf": cmd_perf, "evidence": cmd_evidence}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tdes.worker")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--run-id", default="run-0001")
    ap.add_argument("--until", type=int, default=TOTAL_STEPS)
    ap.add_argument("--crash-at", type=int, default=0)
    ap.add_argument("--frm", type=int, default=1)
    ap.add_argument("--to", type=int, default=TOTAL_STEPS)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args(argv)
    ensure_dirs()
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
