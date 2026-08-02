"""Unit tests for the invariants that do not need a completed run."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tdes.config import LANES, SEQS_PER_STEP, SEQ_LEN, TOTAL_STEPS
from tdes.corpus import build_corpus
from tdes.hashing import chain_hash, hash_obj, merkle_root
from tdes.mixture import MixtureSchedule, enforce_floors
from tdes.model import TinyModel, numeric_gradient_check
from tdes.packing import (ROLE_ID, build_attention_mask, pack_samples,
                          verify_pack)
from tdes.registry import EvalRegistry
from tdes.tokenizer import (EOS, PAD, Tokenizer, normalize, tokenize_document,
                            train_tokenizer)

ZWJ, ZWNJ = "‍", "‌"


def _tiny_tokenizer():
    texts = ["the quiet village measured the trade routes.",
             "def pack_sequence(tokens): return len(tokens)",
             "ज्ञान क्षेत्र विद्यालय पुस्तकालय",
             "கணினி மொழி பயிற்சி ஆய்வு",
             "Theorem 1. We evaluate \\sum_{i=1}^{n} i."]
    return train_tokenizer(texts * 3, vocab_size=340), texts


def _sample(sid, tokens, roles=None, loss=None, structured=False):
    n = len(tokens)
    return {"sample_id": sid, "shard_id": "shard-test-00", "doc_id": "doc-" + sid,
            "lane": "test", "src_start": 0, "src_end": n, "tokens": list(tokens),
            "roles": roles or ["text"] * n, "loss": loss or [1] * n,
            "structured": structured,
            "token_hash": ""}


# --------------------------------------------------------------------------
class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok, self.texts = _tiny_tokenizer()

    def test_deterministic_training(self):
        tok2, _ = _tiny_tokenizer()
        self.assertEqual(self.tok.tokenizer_hash, tok2.tokenizer_hash)
        self.assertEqual(self.tok.merges, tok2.merges)

    def test_roundtrip_lossless(self):
        for t in self.texts + ["क" + ZWJ + "ष", "क्" + ZWNJ + "ष", "mixed मिश्रित 42"]:
            self.assertEqual(self.tok.decode(self.tok.encode(t)), normalize(t))

    def test_indic_zero_width_preserved(self):
        s = "क" + ZWJ + "ष" + ZWNJ + "र"
        self.assertEqual(normalize(s).count(ZWJ), 1)
        self.assertEqual(normalize(s).count(ZWNJ), 1)
        self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_no_nfkc_or_casefold(self):
        spec = self.tok.spec()["normalization"]
        self.assertEqual(spec["form"], "NFC")
        self.assertFalse(spec["nfkc"])
        self.assertFalse(spec["casefold"])
        self.assertFalse(spec["strip_zero_width"])
        self.assertNotEqual(self.tok.encode("ABC"), self.tok.encode("abc"))

    def test_hash_changes_when_spec_changes(self):
        h1 = self.tok.tokenizer_hash
        mutated = Tokenizer(self.tok.merges[:-1], self.tok.vocab_size - 1)
        self.assertNotEqual(h1, mutated.tokenizer_hash)

    def test_special_tokens_are_canonical(self):
        self.assertEqual(PAD, 256)
        self.assertLess(PAD, EOS)
        self.assertNotIn(PAD, self.tok.encode("any text at all"))

    def test_document_loss_flags_follow_roles(self):
        corpus = build_corpus()
        agentic = next(d for d in corpus["train"] if d["lane"] == "agentic")
        enc = tokenize_document(self.tok, agentic)
        self.assertEqual(len(enc["tokens"]), len(enc["loss"]))
        for role, flag in zip(enc["roles"], enc["loss"]):
            if role in ("user", "tool_obs"):
                self.assertEqual(flag, 0, f"context role {role} must not bear loss")
            if role in ("assistant", "tool_call"):
                self.assertEqual(flag, 1)


# --------------------------------------------------------------------------
class TestPacking(unittest.TestCase):
    def setUp(self):
        self.tok, _ = _tiny_tokenizer()

    def _plain(self, n=5):
        rng = np.random.default_rng(0)
        return [_sample(f"s{i}", rng.integers(0, 200, int(rng.integers(20, 90))).tolist())
                for i in range(n)]

    def test_all_policies_satisfy_invariants(self):
        samples = self._plain(8)
        for policy in ("pad_only", "concat_chop", "greedy", "best_fit",
                       "structure_preserving", "long_context"):
            packs = pack_samples(samples, policy, "test")
            self.assertTrue(packs, policy)
            for p in packs:
                self.assertEqual(verify_pack(p), [], f"{policy}/{p['pack_id']}")

    def test_pad_positions_never_bear_loss(self):
        packs = pack_samples(self._plain(3), "pad_only", "test")
        for p in packs:
            for t in range(p["seq_len"]):
                if p["segment_ids"][t] == 0:
                    self.assertEqual(p["input_ids"][t], PAD)
                    self.assertEqual(p["loss_mask"][t], 0)

    def test_position_ids_reset_per_segment(self):
        packs = pack_samples(self._plain(6), "best_fit", "test")
        multi = [p for p in packs if p["n_segments"] > 1]
        self.assertTrue(multi, "expected at least one multi-segment pack")
        for p in multi:
            seen = {}
            for t in range(p["seq_len"]):
                s = p["segment_ids"][t]
                if s == 0:
                    continue
                if s not in seen:
                    self.assertEqual(p["position_ids"][t], 0)
                    seen[s] = 0
                else:
                    seen[s] += 1
                    self.assertEqual(p["position_ids"][t], seen[s])

    def test_attention_mask_is_causal_and_block_diagonal(self):
        seg = [1, 1, 1, 2, 2, 0, 0]
        att = build_attention_mask(seg)
        self.assertTrue(att[2][0] and att[2][1])
        self.assertFalse(att[3][2], "segment 2 must not attend to segment 1")
        self.assertFalse(att[0][1], "no attention to the future")
        self.assertFalse(any(att[5]), "pad attends to nothing")

    def test_loss_never_crosses_a_segment_boundary(self):
        packs = pack_samples(self._plain(6), "structure_preserving", "test")
        for p in packs:
            for t in range(p["seq_len"] - 1):
                if p["segment_ids"][t] != p["segment_ids"][t + 1]:
                    self.assertEqual(p["loss_mask"][t], 0)

    def test_structured_sample_masks_context_roles(self):
        corpus = build_corpus()
        doc = next(d for d in corpus["train"] if d["lane"] == "agentic")
        enc = tokenize_document(self.tok, doc)
        s = _sample("agentic0", enc["tokens"], enc["roles"], enc["loss"], True)
        packs = pack_samples([s], "structure_preserving", "agentic")
        p = packs[0]
        self.assertEqual(verify_pack(p), [])
        for t in range(p["seq_len"] - 1):
            if p["loss_mask"][t]:
                self.assertIn(p["role_ids"][t + 1],
                              {ROLE_ID["assistant"], ROLE_ID["tool_call"],
                               ROLE_ID["think"], ROLE_ID["answer"], ROLE_ID["eos"]})

    def test_concat_chop_marks_boundary_crossings(self):
        packs = pack_samples(self._plain(12), "concat_chop", "test")
        self.assertTrue(sum(p["boundary_crossings"] for p in packs) > 0)
        for p in packs[:-1]:
            self.assertEqual(p["n_real_tokens"], p["seq_len"])

    def test_token_conservation(self):
        for policy in ("pad_only", "greedy", "best_fit", "concat_chop"):
            for p in pack_samples(self._plain(7), policy, "test"):
                self.assertEqual(p["n_real_tokens"] + p["n_pad_tokens"], p["seq_len"])
                self.assertEqual(len(p["input_ids"]), p["seq_len"])

    def test_verify_pack_detects_corruption(self):
        p = pack_samples(self._plain(2), "pad_only", "test")[0]
        bad = dict(p)
        bad["loss_mask"] = list(p["loss_mask"])
        bad["loss_mask"][p["seq_len"] - 1] = 1
        self.assertNotEqual(verify_pack(bad), [])
        bad2 = dict(p)
        bad2["position_ids"] = [0] * p["seq_len"]
        self.assertNotEqual(verify_pack(bad2), [])


# --------------------------------------------------------------------------
class TestModel(unittest.TestCase):
    def test_analytic_gradients_match_finite_differences(self):
        self.assertLess(numeric_gradient_check(), 1e-5)

    def test_attention_mask_isolates_segments(self):
        m = TinyModel(vocab=30, d_model=8, d_ff=12, max_pos=16, seed=1)
        ids = [3, 4, 5, 6, 7, 8, 9, 10]
        seg = [1, 1, 1, 1, 2, 2, 2, 2]
        pos = [0, 1, 2, 3, 0, 1, 2, 3]
        a = m.forward(ids, pos, seg)["logits"][:4].copy()
        ids2 = list(ids)
        ids2[5] = 29
        b = m.forward(ids2, pos, seg)["logits"][:4]
        np.testing.assert_allclose(a, b, atol=0, rtol=0)

    def test_position_ids_change_the_computation(self):
        m = TinyModel(vocab=30, d_model=8, d_ff=12, max_pos=16, seed=2)
        ids = [3, 4, 5, 6]
        seg = [1, 1, 1, 1]
        a = m.forward(ids, [0, 1, 2, 3], seg)["logits"]
        b = m.forward(ids, [4, 5, 6, 7], seg)["logits"]
        self.assertFalse(np.allclose(a, b))

    def test_loss_mask_controls_the_gradient(self):
        m = TinyModel(vocab=30, d_model=8, d_ff=12, max_pos=16, seed=4)
        ids, seg, pos = [3, 4, 5, 6], [1, 1, 1, 1], [0, 1, 2, 3]
        g = m.zero_grads()
        m.accumulate(g, ids, pos, seg, [0, 0, 0, 0], 1.0)
        self.assertTrue(all(float(np.abs(v).sum()) == 0.0 for v in g.values()))
        g2 = m.zero_grads()
        m.accumulate(g2, ids, pos, seg, [1, 1, 1, 0], 1.0)
        self.assertGreater(sum(float(np.abs(v).sum()) for v in g2.values()), 0.0)

    def test_training_reduces_loss_on_a_repeated_batch(self):
        m = TinyModel(vocab=40, d_model=16, d_ff=24, max_pos=16, seed=5)
        ids = [5, 6, 7, 8, 5, 6, 7, 8]
        seg, pos, lm = [1] * 8, list(range(8)), [1] * 7 + [0]
        first, _, _ = m.sequence_loss(ids, pos, seg, lm)
        for _ in range(40):
            g = m.zero_grads()
            m.accumulate(g, ids, pos, seg, lm, 1.0)
            m.step(g, 0.3, 0.9, 1.0)
        last, _, _ = m.sequence_loss(ids, pos, seg, lm)
        self.assertLess(last, first)

    def test_state_roundtrip_is_exact(self):
        m = TinyModel(vocab=20, d_model=8, d_ff=8, max_pos=8, seed=6)
        h = m.param_hash()
        m2 = TinyModel(vocab=20, d_model=8, d_ff=8, max_pos=8, seed=7)
        m2.load_state_dict(m.state_dict())
        self.assertEqual(h, m2.param_hash())


# --------------------------------------------------------------------------
class TestMixture(unittest.TestCase):
    def test_alloc_sums_to_batch_size_every_step(self):
        mix = MixtureSchedule()
        for s in range(1, TOTAL_STEPS + 1):
            alloc = mix.alloc(s)
            self.assertEqual(sum(alloc.values()), SEQS_PER_STEP, f"step {s}")
            self.assertTrue(all(v >= 0 for v in alloc.values()))

    def test_alloc_is_a_pure_function_of_step(self):
        a = MixtureSchedule()
        b = MixtureSchedule()
        # b is queried in reverse order: caching must not change the answer
        back = {s: b.alloc(s) for s in range(TOTAL_STEPS, 0, -1)}
        for s in range(1, TOTAL_STEPS + 1):
            self.assertEqual(a.alloc(s), back[s])

    def test_protected_floors_survive_compilation(self):
        mix = MixtureSchedule()
        for st in mix.stages:
            for lane, floor in st["protected_floors"].items():
                self.assertGreaterEqual(st["effective_mixture"][lane], floor - 1e-9)
            self.assertAlmostEqual(sum(st["effective_mixture"].values()), 1.0, places=9)

    def test_floor_enforcement_raises_starved_lane(self):
        w = {l: 0.0 for l in LANES}
        w["general_web"] = 1.0
        eff = enforce_floors(w, {"indic": 0.15, "agentic": 0.05})
        self.assertGreaterEqual(eff["indic"], 0.15 - 1e-9)
        self.assertGreaterEqual(eff["agentic"], 0.05 - 1e-9)
        self.assertAlmostEqual(sum(eff.values()), 1.0, places=9)

    def test_override_changes_the_schedule_hash(self):
        a = MixtureSchedule().to_json()["schedule_hash"]
        b = MixtureSchedule({"reasoning": 0.22, "general_web": 0.20}) \
            .to_json()["schedule_hash"]
        self.assertNotEqual(a, b)

    def test_anneal_reserve_is_withheld_then_released(self):
        mix = MixtureSchedule()
        early = mix.reserve_fraction("reasoning", 1)
        late = mix.reserve_fraction("reasoning", TOTAL_STEPS)
        self.assertGreater(early, 0.0)
        self.assertEqual(late, 0.0)

    def test_warmup_blends_between_stages(self):
        mix = MixtureSchedule()
        boundary = None
        for s in range(1, TOTAL_STEPS + 1):
            if mix.stage_for_step(s)["stage"] == "reasoning-heavy-midtrain":
                boundary = s
                break
        self.assertIsNotNone(boundary)
        w_first = mix.weights_at_step(boundary)
        w_later = mix.weights_at_step(boundary + 5)
        self.assertNotEqual([round(v, 6) for v in w_first.values()],
                            [round(v, 6) for v in w_later.values()])


# --------------------------------------------------------------------------
class TestHashingAndFirewall(unittest.TestCase):
    def test_chain_detects_tampering(self):
        h0 = chain_hash("0" * 64, {"a": 1})
        h1 = chain_hash(h0, {"a": 2})
        self.assertNotEqual(h1, chain_hash(h0, {"a": 3}))
        self.assertNotEqual(h1, chain_hash("0" * 64, {"a": 2}))

    def test_merkle_root_changes_with_any_leaf(self):
        leaves = [hash_obj(i) for i in range(5)]
        r1 = merkle_root(leaves)
        leaves[2] = hash_obj("x")
        self.assertNotEqual(r1, merkle_root(leaves))

    def test_registry_detects_contaminated_document(self):
        tok, _ = _tiny_tokenizer()
        corpus = build_corpus()
        reg = EvalRegistry()
        for d in corpus["test"]:
            reg.register(d, tok, "never_train")
        bad = next(d for d in corpus["rejected_seed"]
                   if "contaminated" in d["doc_id"])
        self.assertTrue(reg.scan_text(bad["text"]))
        clean = next(d for d in corpus["train"] if d["lane"] == "math_science")
        self.assertFalse(reg.scan_text(clean["text"]))

    def test_registry_detects_smuggled_eval_document(self):
        tok, _ = _tiny_tokenizer()
        corpus = build_corpus()
        reg = EvalRegistry()
        for d in corpus["test"]:
            reg.register(d, tok, "never_train")
        smuggled = next(d for d in corpus["rejected_seed"]
                        if "evalsmuggle" in d["doc_id"])
        self.assertTrue(reg.is_registered_text(smuggled["text"]) or
                        reg.scan_text(smuggled["text"]))

    def test_token_level_scan_finds_overlap(self):
        tok, _ = _tiny_tokenizer()
        corpus = build_corpus()
        reg = EvalRegistry()
        for d in corpus["test"]:
            reg.register(d, tok, "never_train")
        ev_tokens = tokenize_document(tok, corpus["test"][1])["tokens"]
        self.assertTrue(reg.scan_tokens(ev_tokens))
        self.assertFalse(reg.scan_tokens(list(range(60, 90))))


if __name__ == "__main__":
    unittest.main()
