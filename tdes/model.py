"""A tiny but real causal transformer block, in float64 numpy.

It is deliberately small, but it is *not* a stub: it actually consumes the
attention mask (block-diagonal by packed segment), the position ids (learned
positional embeddings indexed by the per-segment position) and the loss mask
(only masked positions contribute to the gradient).  If the data system got any
of those wrong, this model would train on the wrong thing - which is exactly
what the invariant tests check.

float64 throughout so that replay is bit-identical.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .hashing import hash_obj

NEG = -1e30


class TinyModel:
    def __init__(self, vocab: int, d_model: int, d_ff: int, max_pos: int, seed: int):
        rng = np.random.default_rng(seed)
        s = 0.08
        self.cfg = {"vocab": vocab, "d_model": d_model, "d_ff": d_ff,
                    "max_pos": max_pos, "seed": seed}
        self.p: Dict[str, np.ndarray] = {
            "E": rng.normal(0, s, (vocab, d_model)),
            "P": rng.normal(0, s, (max_pos, d_model)),
            "Wq": rng.normal(0, s, (d_model, d_model)),
            "Wk": rng.normal(0, s, (d_model, d_model)),
            "Wv": rng.normal(0, s, (d_model, d_model)),
            "Wo": rng.normal(0, s, (d_model, d_model)),
            "W1": rng.normal(0, s, (d_model, d_ff)),
            "W2": rng.normal(0, s, (d_ff, d_model)),
            "Wu": rng.normal(0, s, (d_model, vocab)),
            "bu": np.zeros(vocab),
        }
        self.p = {k: v.astype(np.float64) for k, v in self.p.items()}
        self.mom: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.p.items()}

    # ------------------------------------------------------------ plumbing --
    def zero_grads(self) -> Dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.p.items()}

    @staticmethod
    def attention_mask(seg: Sequence[int]) -> np.ndarray:
        s = np.asarray(seg)
        allow = (s[:, None] == s[None, :]) & (s[:, None] != 0)
        causal = np.tril(np.ones((len(s), len(s)), dtype=bool))
        m = allow & causal
        np.fill_diagonal(m, True)   # keep softmax finite on pad rows
        return m

    # ------------------------------------------------------------- forward --
    def forward(self, ids: Sequence[int], pos: Sequence[int], seg: Sequence[int]):
        p = self.p
        x = np.asarray(ids, dtype=np.int64)
        pi = np.asarray(pos, dtype=np.int64)
        d = self.cfg["d_model"]
        h0 = p["E"][x] + p["P"][pi]
        q, k, v = h0 @ p["Wq"], h0 @ p["Wk"], h0 @ p["Wv"]
        s = (q @ k.T) / math.sqrt(d)
        mask = self.attention_mask(seg)
        s = np.where(mask, s, NEG)
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        a = e / e.sum(axis=-1, keepdims=True)
        c = a @ v
        o = c @ p["Wo"]
        h1 = h0 + o
        z = h1 @ p["W1"]
        relu = np.maximum(z, 0.0)
        h2 = h1 + relu @ p["W2"]
        logits = h2 @ p["Wu"] + p["bu"]
        return {"x": x, "pi": pi, "h0": h0, "q": q, "k": k, "v": v, "a": a,
                "c": c, "h1": h1, "z": z, "relu": relu, "h2": h2,
                "logits": logits, "mask": mask}

    @staticmethod
    def _token_losses(logits: np.ndarray, ids: np.ndarray,
                      loss_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = logits.shape[0]
        m = logits.max(axis=-1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
        tgt = np.zeros(L, dtype=np.int64)
        tgt[:-1] = ids[1:]
        picked = logits[np.arange(L), tgt]
        losses = lse - picked
        return losses * loss_mask, tgt

    def sequence_loss(self, ids, pos, seg, loss_mask) -> Tuple[float, np.ndarray, int]:
        """Forward-only loss (used by the OPUS proxy and validation eval)."""
        cache = self.forward(ids, pos, seg)
        lm = np.asarray(loss_mask, dtype=np.float64)
        losses, _ = self._token_losses(cache["logits"], cache["x"], lm)
        n = int(lm.sum())
        return (float(losses.sum() / n) if n else 0.0), losses, n

    # ------------------------------------------------------------ backward --
    def accumulate(self, grads: Dict[str, np.ndarray], ids, pos, seg, loss_mask,
                   scale: float) -> Tuple[float, np.ndarray, int]:
        p = self.p
        d = self.cfg["d_model"]
        cache = self.forward(ids, pos, seg)
        logits, x = cache["logits"], cache["x"]
        lm = np.asarray(loss_mask, dtype=np.float64)
        n = int(lm.sum())
        losses, tgt = self._token_losses(logits, x, lm)
        if n == 0:
            return 0.0, losses, 0

        L, V = logits.shape
        mx = logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits - mx)
        prob = ex / ex.sum(axis=-1, keepdims=True)
        dlogits = prob.copy()
        dlogits[np.arange(L), tgt] -= 1.0
        dlogits *= (lm * (scale / n))[:, None]

        grads["Wu"] += cache["h2"].T @ dlogits
        grads["bu"] += dlogits.sum(axis=0)
        dh2 = dlogits @ p["Wu"].T

        dh1 = dh2.copy()
        drelu = dh2 @ p["W2"].T
        grads["W2"] += cache["relu"].T @ dh2
        dz = drelu * (cache["z"] > 0)
        grads["W1"] += cache["h1"].T @ dz
        dh1 += dz @ p["W1"].T

        dh0 = dh1.copy()
        do = dh1
        grads["Wo"] += cache["c"].T @ do
        dc = do @ p["Wo"].T

        a, v = cache["a"], cache["v"]
        da = dc @ v.T
        dv = a.T @ dc
        ds = a * (da - (da * a).sum(axis=-1, keepdims=True))
        ds = np.where(cache["mask"], ds, 0.0) / math.sqrt(d)
        dq = ds @ cache["k"]
        dk = ds.T @ cache["q"]

        grads["Wq"] += cache["h0"].T @ dq
        grads["Wk"] += cache["h0"].T @ dk
        grads["Wv"] += cache["h0"].T @ dv
        dh0 += dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T

        np.add.at(grads["E"], cache["x"], dh0)
        np.add.at(grads["P"], cache["pi"], dh0)
        return float(losses.sum() / n), losses, n

    # ------------------------------------------------------------ optimizer --
    def step(self, grads: Dict[str, np.ndarray], lr: float, momentum: float,
             clip: float) -> float:
        total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        factor = 1.0 if total <= clip or total == 0 else clip / total
        for k in self.p:
            g = grads[k] * factor
            self.mom[k] = momentum * self.mom[k] + g
            self.p[k] -= lr * self.mom[k]
        return total

    # ---------------------------------------------------------------- state --
    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg,
            "params": {k: v.tolist() for k, v in self.p.items()},
            "momentum": {k: v.tolist() for k, v in self.mom.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.cfg = sd["cfg"]
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in sd["params"].items()}
        self.mom = {k: np.asarray(v, dtype=np.float64) for k, v in sd["momentum"].items()}

    def param_hash(self) -> str:
        return hash_obj({k: np.round(v, 12).tolist() for k, v in sorted(self.p.items())})

    def clone(self) -> "TinyModel":
        m = TinyModel(**self.cfg)
        m.p = {k: v.copy() for k, v in self.p.items()}
        m.mom = {k: v.copy() for k, v in self.mom.items()}
        return m


def numeric_gradient_check(seed: int = 3) -> float:
    """Finite-difference check of the hand-written backward pass.

    Probes the largest-magnitude entry of each parameter (a random entry is
    often exactly zero, where a relative comparison is meaningless).
    """
    m = TinyModel(vocab=17, d_model=6, d_ff=8, max_pos=12, seed=seed)
    ids = [3, 11, 5, 2, 9, 14, 1, 7, 0, 0]
    seg = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0]
    pos = [0, 1, 2, 3, 0, 1, 2, 3, 0, 0]
    lm = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
    g = m.zero_grads()
    m.accumulate(g, ids, pos, seg, lm, 1.0)
    worst = 0.0
    # the largest gradient entries of this tiny model are ~1e-6, so a smaller
    # step drowns the central difference in float64 cancellation noise
    eps = 1e-4
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wu", "bu", "E", "P"):
        arr = m.p[name]
        idx = np.unravel_index(int(np.argmax(np.abs(g[name]))), arr.shape)
        ana = float(g[name][idx])
        if abs(ana) < 1e-9:
            continue
        orig = arr[idx]
        arr[idx] = orig + eps
        lp, _, _ = m.sequence_loss(ids, pos, seg, lm)
        arr[idx] = orig - eps
        ln, _, _ = m.sequence_loss(ids, pos, seg, lm)
        arr[idx] = orig
        num = (lp - ln) / (2 * eps)
        worst = max(worst, abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
    return worst
