"""Checkpoints that bind model state to data state.

A checkpoint here is incomplete unless it carries the data position:
``next_step``, the per-ledger offsets, the per-ledger hash-chain heads and the
dataloader/stream state.  Restoring a checkpoint therefore restores *both*
halves of the experiment definition.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Dict, Optional

import numpy as np

from .config import PATHS, config_fingerprint
from .hashing import hash_obj, sha256_hex


def ckpt_dir(branch: str) -> str:
    d = os.path.join(PATHS["checkpoints"], branch)
    os.makedirs(d, exist_ok=True)
    return d


def ckpt_id(branch: str, step: int) -> str:
    return f"ckpt-{branch}-{step:04d}"


def ckpt_path(branch: str, step: int) -> str:
    return os.path.join(ckpt_dir(branch), f"{ckpt_id(branch, step)}.json")


def _enc(arrs: Dict[str, np.ndarray]) -> Dict[str, dict]:
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                "b64": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode()}
            for k, v in sorted(arrs.items())}


def _dec(d: Dict[str, dict]) -> Dict[str, np.ndarray]:
    return {k: np.frombuffer(base64.b64decode(v["b64"]),
                             dtype=np.dtype(v["dtype"])).reshape(v["shape"]).copy()
            for k, v in d.items()}


def tensor_hash(arrs: Dict[str, np.ndarray]) -> str:
    h = {k: sha256_hex(np.ascontiguousarray(v).tobytes())
         for k, v in sorted(arrs.items())}
    return hash_obj(h)


def save_checkpoint(branch: str, run_id: str, step: int, next_step: int,
                    model, stream_state: dict, ledgers, extra: dict) -> dict:
    doc = {
        "schema": "tdes-checkpoint/1",
        "checkpoint_id": ckpt_id(branch, step),
        "branch_id": branch,
        "run_id": run_id,
        "global_step": step,
        "next_step": next_step,
        "model_cfg": model.cfg,
        "params": _enc(model.p),
        "momentum": _enc(model.mom),
        "param_hash": tensor_hash(model.p),
        "optimizer_hash": tensor_hash(model.mom),
        "stream_state": stream_state,
        "ledger_offsets": ledgers.offsets(),
        "ledger_heads": ledgers.heads(),
        "config_fingerprint": config_fingerprint(),
    }
    doc.update(extra)
    doc["checkpoint_hash"] = hash_obj({k: v for k, v in doc.items()
                                       if k not in ("params", "momentum",
                                                    "checkpoint_hash")})
    path = ckpt_path(branch, step)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return doc


def load_checkpoint(branch: str, step: int) -> dict:
    with open(ckpt_path(branch, step), "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    recomputed = hash_obj({k: v for k, v in doc.items()
                           if k not in ("params", "momentum", "checkpoint_hash")})
    if recomputed != doc["checkpoint_hash"]:
        raise ValueError(f"checkpoint {doc['checkpoint_id']} was mutated")
    return doc


def restore_model(model, doc: dict) -> None:
    model.p = _dec(doc["params"])
    model.mom = _dec(doc["momentum"])
    model.cfg = doc["model_cfg"]
    if tensor_hash(model.p) != doc["param_hash"]:
        raise ValueError("param hash mismatch after restore")


def latest_checkpoint(branch: str, at_or_before: Optional[int] = None) -> Optional[dict]:
    d = ckpt_dir(branch)
    best = None
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        step = int(name.rsplit("-", 1)[1].split(".")[0])
        if at_or_before is not None and step > at_or_before:
            continue
        if best is None or step > best:
            best = step
    return None if best is None else load_checkpoint(branch, best)


def list_checkpoints(branch: str) -> list:
    d = ckpt_dir(branch)
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            out.append(int(name.rsplit("-", 1)[1].split(".")[0]))
    return sorted(out)
