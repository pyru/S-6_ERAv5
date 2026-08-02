"""Execution log writer.

Emits a human-readable, greppable run.log *and* a structured mirror
(run_events.jsonl) so the evidence builder can re-derive what happened from
disk instead of from memory.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from typing import Any

from .config import PATHS
from .hashing import canon


class RunLog:
    def __init__(self, path: str | None = None, echo: bool = True, append: bool = False):
        self.path = path or PATHS["run_log"]
        self.events_path = os.path.join(os.path.dirname(self.path), "run_events.jsonl")
        self.echo = echo
        mode = "a" if append else "w"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, mode, encoding="utf-8")
        self._ev = open(self.events_path, mode, encoding="utf-8")
        self._n = 0

    # ------------------------------------------------------------ internals --
    def _stamp(self) -> str:
        return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    def _write(self, line: str) -> None:
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.echo:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _emit(self, level: str, event: str, fields: dict) -> None:
        self._n += 1
        payload = {"n": self._n, "ts": self._stamp(), "level": level,
                   "event": event, "fields": fields}
        self._ev.write(canon(payload) + "\n")
        self._ev.flush()
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        self._write(f"{payload['ts']} [{level}] {event}" + (f" {extra}" if extra else ""))

    # --------------------------------------------------------------- public --
    def section(self, title: str) -> None:
        bar = "=" * 78
        self._write("")
        self._write(bar)
        self._write(f"== {title}")
        self._write(bar)
        self._n += 1
        self._ev.write(canon({"n": self._n, "ts": self._stamp(), "level": "SECTION",
                              "event": title, "fields": {}}) + "\n")
        self._ev.flush()

    def info(self, event: str, **fields: Any) -> None:
        self._emit("INFO", event, fields)

    def event(self, event: str, **fields: Any) -> None:
        self._emit("EVENT", event, fields)

    def ok(self, event: str, **fields: Any) -> None:
        self._emit("PASS", event, fields)

    def fail(self, event: str, **fields: Any) -> None:
        self._emit("FAIL", event, fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._emit("WARN", event, fields)

    def check(self, condition: bool, event: str, **fields: Any) -> bool:
        (self.ok if condition else self.fail)(event, **fields)
        return bool(condition)

    def close(self) -> None:
        self._fh.close()
        self._ev.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list, tuple)):
        return canon(v)
    return str(v)


def read_events(path: str | None = None) -> list:
    p = path or os.path.join(PATHS["art"], "run_events.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
