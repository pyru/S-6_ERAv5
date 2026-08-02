#!/usr/bin/env python
"""One command that runs the complete TDES demonstration.

    python run_demo.py

Regenerates submission_artifacts/ from scratch:

    corpus -> frozen tokenizer -> immutable shards + manifests -> eval firewall
    -> mixture timeline -> packing -> training with OPUS + ledgers
    -> checkpoint -> REAL process crash -> resume -> replay -> fork
    -> audit -> invariant tests -> performance -> evidence bundle

Each phase runs in its own OS process, so the crash is a genuine abrupt
process death and recovery has to come from durable state on disk.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tdes.config import (CHECKPOINT_EVERY, CRASH_AT_STEP, FORK_FROM_STEP,  # noqa: E402
                         FORK_STEPS, PATHS, REPLAY_FROM, REPLAY_TO,
                         TOTAL_STEPS, ensure_dirs)
from tdes.runlog import RunLog  # noqa: E402
from tdes.worker import CRASH_EXIT_CODE  # noqa: E402

RUN_ID = "run-0001"


def _force_rm(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clean() -> None:
    if os.path.exists(PATHS["art"]):
        shutil.rmtree(PATHS["art"], onerror=_force_rm)
    ensure_dirs()


EXIT_CODES: dict = {}


def phase(name: str, argv: list, expect_code: int = 0,
          allow_fail: bool = False) -> float:
    print(f"\n>>> {name}: python -m tdes.worker {' '.join(argv)}")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "tdes.worker"] + argv,
                          cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                         "PYTHONUTF8": "1"})
    dt = time.perf_counter() - t0
    EXIT_CODES[name] = proc.returncode
    if proc.returncode != expect_code and not allow_fail:
        raise SystemExit(f"phase '{name}' exited {proc.returncode}, "
                         f"expected {expect_code}")
    return dt


def run_tests(name: str, out_file: str) -> dict:
    print(f"\n>>> invariant test suite ({name})")
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(ROOT, "tests"), top_level_dir=ROOT)
    buf = io.StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    out = buf.getvalue()
    print(out[-4000:] if len(out) > 4000 else out)
    doc = {
        "pass_name": name,
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "failure_details": [f"{t}: {(m.strip().splitlines() or [''])[-1]}"
                            for t, m in result.failures + result.errors],
        "output": out,
    }
    os.makedirs(PATHS["reports"], exist_ok=True)
    with open(os.path.join(PATHS["reports"], out_file), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return doc


def main() -> int:
    t_all = time.perf_counter()
    clean()

    log = RunLog(append=False)
    log.section("TDES - TRAINING DATA EXECUTION SYSTEM (V5, SESSION 6)")
    log.info("demo_started", run_id=RUN_ID, python=sys.version.split()[0],
             total_steps=TOTAL_STEPS, checkpoint_every=CHECKPOINT_EVERY,
             crash_at=CRASH_AT_STEP,
             replay_interval=f"{REPLAY_FROM}..{REPLAY_TO}",
             fork_from=FORK_FROM_STEP)
    log.close()

    timings = {}
    timings["prepare"] = phase("prepare", ["prepare", "--run-id", RUN_ID])
    timings["train_until_crash"] = phase(
        "train (crashes on purpose)",
        ["train", "--run-id", RUN_ID, "--until", str(TOTAL_STEPS),
         "--crash-at", str(CRASH_AT_STEP)], expect_code=CRASH_EXIT_CODE)
    timings["resume"] = phase("resume", ["resume", "--run-id", RUN_ID,
                                         "--until", str(TOTAL_STEPS)])
    timings["replay"] = phase("replay", ["replay", "--run-id", RUN_ID,
                                         "--frm", str(REPLAY_FROM),
                                         "--to", str(REPLAY_TO)])
    timings["fork"] = phase("fork", ["fork", "--run-id", RUN_ID,
                                     "--steps", str(FORK_STEPS)])
    timings["audit"] = phase("audit", ["audit", "--run-id", RUN_ID],
                             allow_fail=True)

    with open(os.path.join(PATHS["reports"], "phase_timings.json"), "w",
              encoding="utf-8") as fh:
        json.dump({k: round(v, 4) for k, v in timings.items()}, fh, indent=1,
                  sort_keys=True)
    timings["performance"] = phase("performance", ["perf", "--run-id", RUN_ID],
                                   allow_fail=True)

    # pass 1 feeds the evidence bundle; the evidence checks themselves skip here
    t0 = time.perf_counter()
    tests = run_tests("pre-evidence", "test_results.json")
    timings["tests"] = time.perf_counter() - t0

    timings["evidence"] = phase("evidence", ["evidence", "--run-id", RUN_ID],
                                allow_fail=True)

    # pass 2 re-runs everything now that evidence.json exists
    t0 = time.perf_counter()
    final = run_tests("post-evidence", "test_results_final.json")
    timings["tests_final"] = time.perf_counter() - t0

    with open(os.path.join(PATHS["reports"], "phase_timings.json"), "w",
              encoding="utf-8") as fh:
        json.dump({**{k: round(v, 4) for k, v in timings.items()},
                   "total_demo_seconds": round(time.perf_counter() - t_all, 4)},
                  fh, indent=1, sort_keys=True)

    with open(PATHS["evidence_json"], "r", encoding="utf-8") as fh:
        ev = json.load(fh)

    print("\n" + "=" * 78)
    print(f"OVERALL: {ev['overall_result']}  "
          f"({ev['requirements_passed']}/{ev['requirements_total']} requirements)")
    for q in ev["requirements"]:
        print(f"  [{q['result']}] {q['requirement']:<52} "
              f"{q['checks_passed']}/{q['checks_total']} checks")
    print(f"  tests (pre-evidence):  {tests['total']} run, {tests['failures']} "
          f"failures, {tests['errors']} errors, {tests['skipped']} skipped")
    print(f"  tests (post-evidence): {final['total']} run, {final['failures']} "
          f"failures, {final['errors']} errors, {final['skipped']} skipped")
    for d in final["failure_details"]:
        print(f"    ! {d}")
    print(f"  artifacts: {PATHS['art']}")
    print(f"  wall time: {time.perf_counter() - t_all:.1f}s")
    print("=" * 78)
    return 0 if ev["overall_result"] == "PASS" and not final["failures"] \
        and not final["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
