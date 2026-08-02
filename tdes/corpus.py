"""Deterministic synthetic corpus (Sessions 3 + 4 contracts).

No network, no external data: every document is generated from a seeded RNG so
the whole pipeline is byte-reproducible.  Each document carries the provenance,
license, held-out status, capability tags and cleaning lineage that Session 3
and Session 4 require, plus deliberately-bad documents so the admission gate
and evaluation firewall have something real to reject.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .config import CLEANING_PIPELINE_VERSION, CORPUS_SEED, PATHS
from .hashing import hash_obj, hash_text, stable_rng

CANARY = "CANARY-TDES-EVAL-7f3a91"

# ------------------------------------------------------------- word banks --
WEB_SUBJ = ["the harbour city", "a quiet village", "the research team", "the monsoon",
            "a small library", "the night train", "the river delta", "the old market"]
WEB_VERB = ["reshaped", "documented", "measured", "connected", "outlasted", "carried",
            "revealed", "sheltered"]
WEB_OBJ = ["the trade routes", "the seasonal rainfall", "a century of records",
           "two hundred families", "the coastal wetlands", "an unusual pattern",
           "the printing presses", "the migration paths"]
WEB_TAIL = ["Local records confirm the account.", "The findings were later revised.",
            "Historians still debate the timeline.", "The effect persisted for decades.",
            "A follow-up survey is planned.", "The archive remains open to the public."]

CODE_FN = ["normalize_shard", "pack_sequence", "verify_manifest", "compute_loss_mask",
           "resume_stream", "merge_ledgers", "score_candidate", "hash_tokens"]
CODE_ARG = ["tokens", "manifest", "batch", "offset", "policy", "spans"]

MATH_LHS = ["\\sum_{i=1}^{n} i", "\\int_0^1 x^2 dx", "e^{i\\pi}", "\\frac{d}{dx} x^3",
            "\\lim_{n\\to\\infty} (1+1/n)^n", "\\binom{n}{k}"]
MATH_RHS = ["\\frac{n(n+1)}{2}", "\\frac{1}{3}", "-1", "3x^2", "e",
            "\\frac{n!}{k!(n-k)!}"]

HI_WORDS = ["ज्ञान", "क्षेत्र", "विद्यालय", "पुस्तकालय", "संस्कृति", "प्रशिक्षण",
            "विश्लेषण", "अभियांत्रिकी", "त्रिकोण", "श्रेणी", "निर्देश"]
HI_TAIL = ["यह जानकारी सत्यापित है।", "विवरण अभिलेख में दर्ज है।",
           "अध्ययन जारी है।", "परिणाम अगले चरण में देखे जाएंगे।"]
# ZWNJ (U+200C) / ZWJ (U+200D) bearing forms - must survive normalization intact
HI_ZW = ["क‍ष", "क्‌ष",
         "ज्‍ञ", "र‍ु"]
TA_WORDS = ["கணினி", "மொழி", "பயிற்சி", "ஆய்வு", "தரவு", "நூலகம்", "கட்டமைப்பு"]
TA_TAIL = ["இந்த தகவல் சரிபார்க்கப்பட்டது.", "ஆய்வு தொடர்கிறது."]

AGENT_TASKS = ["find the shard manifest for lane {lane}",
               "check whether run {rid} skipped a batch",
               "summarise the mixture drift for stage {st}",
               "verify the tokenizer hash of the frozen vocabulary"]
TOOLS = ["ledger.query", "manifest.read", "shard.stat", "mixture.report"]

REASON_Q = ["If a batch holds {a} sequences of {b} tokens, how many token positions is that?",
            "A shard has {a} documents averaging {b} tokens. How many tokens total?",
            "If {a} percent of {b} positions are padding, how many are useful?"]


def _cleaning_hash(stage: str) -> str:
    return hash_obj({"pipeline": CLEANING_PIPELINE_VERSION, "stage": stage,
                     "ops": ["nfc", "boilerplate_strip", "dedup_minhash",
                             "pii_screen", "lang_id", "contamination_scan"]})[:32]


def _doc(doc_id, source_id, lane, split, segments, lang, script, **kw) -> dict:
    text = "".join(s["text"] for s in segments)
    d = {
        "doc_id": doc_id,
        "source_id": source_id,
        "lane": lane,
        "split": split,
        "segments": segments,
        "text": text,
        "text_hash": hash_text(text),
        "lang": lang,
        "script": script,
        "license_tier": kw.get("license_tier", "permissive"),
        "provenance_tier": kw.get("provenance_tier", "A"),
        "holdout": kw.get("holdout", False),
        "capability_tags": kw.get("capability_tags", [lane]),
        "cleaning_pipeline_hash": kw.get("cleaning_pipeline_hash", _cleaning_hash(lane)),
        "dedup_status": kw.get("dedup_status", "unique"),
        "pii_status": kw.get("pii_status", "clean"),
        "lang_validated": kw.get("lang_validated", True),
        "benchmark_id": kw.get("benchmark_id", ""),
        "notes": kw.get("notes", ""),
    }
    return d


# ------------------------------------------------------------- generators --
def _seg(role, text, loss):
    return {"role": role, "text": text, "loss": loss}


def _gen_web(rng, i):
    n = rng.randint(4, 7)
    body = " ".join(f"{rng.choice(WEB_SUBJ).capitalize()} {rng.choice(WEB_VERB)} "
                    f"{rng.choice(WEB_OBJ)}. {rng.choice(WEB_TAIL)}" for _ in range(n))
    return [_seg("text", body, True)]


def _gen_code(rng, i):
    fn = rng.choice(CODE_FN)
    args = ", ".join(rng.sample(CODE_ARG, rng.randint(1, 3)))
    body = (f"def {fn}({args}):\n"
            f"    \"\"\"{fn.replace('_', ' ')}.\"\"\"\n"
            f"    total = 0\n"
            f"    for i, item in enumerate({args.split(',')[0]}):\n"
            f"        if item is None:\n"
            f"            continue\n"
            f"        total += len(item) * {rng.randint(2, 9)}\n"
            f"    assert total >= 0, \"invariant broken\"\n"
            f"    return total\n")
    return [_seg("text", body, True)]


def _gen_math(rng, i):
    k = rng.randint(0, len(MATH_LHS) - 1)
    body = (f"Theorem {i}. We evaluate {MATH_LHS[k]}.\n"
            f"Proof. Expand the definition and collect terms.\n"
            f"Therefore {MATH_LHS[k]} = {MATH_RHS[k]}.\n"
            f"Hence the identity holds for all admissible n. QED\n")
    return [_seg("text", body, True)]


def _gen_indic(rng, i):
    if i % 3 == 2:
        words = [rng.choice(TA_WORDS) for _ in range(rng.randint(6, 10))]
        body = " ".join(words) + " " + rng.choice(TA_TAIL)
        return [_seg("text", body, True)], "ta", "Taml"
    words = [rng.choice(HI_WORDS) for _ in range(rng.randint(6, 10))]
    words.insert(rng.randint(0, len(words)), rng.choice(HI_ZW))
    body = " ".join(words) + " " + rng.choice(HI_TAIL)
    return [_seg("text", body, True)], "hi", "Deva"


def _gen_agentic(rng, i):
    task = rng.choice(AGENT_TASKS).format(lane=rng.choice(["code", "indic"]),
                                          rid=rng.randint(100, 999),
                                          st=rng.randint(0, 2))
    tool = rng.choice(TOOLS)
    return [
        _seg("user", f"Please {task}.", False),
        _seg("assistant", f"I will call {tool} to resolve this.", True),
        _seg("tool_call", json.dumps({"tool": tool, "args": {"limit": rng.randint(2, 40)}},
                                     sort_keys=True), True),
        _seg("tool_obs", json.dumps({"rows": rng.randint(1, 9), "status": "ok"},
                                    sort_keys=True), False),
        _seg("assistant", f"The {tool} result confirms the request; nothing was skipped.",
             True),
    ]


def _gen_reasoning(rng, i):
    a, b = rng.randint(2, 16), rng.randint(64, 512)
    q = rng.choice(REASON_Q).format(a=a, b=b)
    return [
        _seg("user", q, False),
        _seg("think", f"Break it down. First identify {a} and {b}. "
                      f"Multiply {a} by {b} to get {a * b}. Check the units are positions.",
             True),
        _seg("answer", f"The answer is {a * b}.", True),
    ]


LANE_GEN = {
    "general_web": _gen_web,
    "code": _gen_code,
    "math_science": _gen_math,
    "agentic": _gen_agentic,
    "reasoning": _gen_reasoning,
}

LANE_DOCS = {
    "general_web": 30,
    "code": 26,
    "math_science": 24,
    "indic": 28,
    "agentic": 22,
    "reasoning": 24,
}


def build_corpus() -> Dict[str, List[dict]]:
    """Returns {'train': [...], 'validation': [...], 'test': [...]}."""
    rng = stable_rng("corpus", CORPUS_SEED)
    train: List[dict] = []

    for lane in ("general_web", "code", "math_science", "indic", "agentic", "reasoning"):
        n = LANE_DOCS[lane]
        for i in range(n):
            did = f"doc-{lane}-{i:03d}"
            src = f"src-{lane}-{i % 4}"
            if lane == "indic":
                segs, lang, script = _gen_indic(rng, i)
            else:
                segs, lang, script = LANE_GEN[lane](rng, i), "en", "Latn"
            tier = "A" if i % 5 else "B"
            train.append(_doc(did, src, lane, "train", segs, lang, script,
                              provenance_tier=tier,
                              license_tier="permissive" if i % 7 else "cc_by",
                              capability_tags=[lane] + (["long_context"] if i % 9 == 0 else [])))

    # ------------------------------------------------ evaluation / test set --
    test: List[dict] = []
    for i in range(6):
        body = (f"{CANARY} Benchmark item {i}. "
                f"Question: which shard produced token span {i * 97}? "
                f"Answer: shard-general_web-{i % 3:02d}. "
                f"This exact string must never appear in a gradient-bearing batch.")
        test.append(_doc(f"doc-eval-{i:03d}", "bench-tdes", "eval", "test",
                         [_seg("text", body, False)], "en", "Latn",
                         holdout=True, benchmark_id=f"TDES-BENCH-v1/item{i}",
                         license_tier="permissive",
                         capability_tags=["benchmark"]))

    # validation text is deliberately disjoint from every training sentence bank,
    # so a held-out/train overlap can only appear if the firewall actually leaks
    val_bank = ["Probe sentences avoid every training phrase bank on purpose.",
                "Held-out prose exists so drift can be measured, not memorised.",
                "Nothing below should ever reach an optimizer update.",
                "Evaluation reads are logged; gradient reads are refused.",
                "Disjoint vocabulary keeps overlap detection meaningful."]
    validation: List[dict] = []
    for i in range(5):
        body = ("Validation probe %d. The dataloader may read this for evaluation "
                "but must never apply gradients to it. %s" %
                (i, " ".join(val_bank[(i + k) % len(val_bank)] for k in range(3))))
        validation.append(_doc(f"doc-val-{i:03d}", "held-out-val", "validation",
                               "validation", [_seg("text", body, False)], "en", "Latn",
                               holdout=True, capability_tags=["validation"]))

    # ------------------------------------------- deliberately bad documents --
    bad = []
    # 1. exact duplicate of an admitted doc
    dup = dict(train[3])
    dup = _doc("doc-bad-dup-000", dup["source_id"], dup["lane"], "train",
               dup["segments"], dup["lang"], dup["script"],
               dedup_status="duplicate", notes="exact duplicate of " + train[3]["doc_id"])
    bad.append(dup)
    # 2. non-commercial license
    bad.append(_doc("doc-bad-license-000", "src-scraped-9", "general_web", "train",
                    _gen_web(rng, 0), "en", "Latn",
                    license_tier="noncommercial", notes="license tier not admissible"))
    # 3. missing cleaning lineage
    bad.append(_doc("doc-bad-lineage-000", "src-unknown-1", "code", "train",
                    _gen_code(rng, 0), "en", "Latn",
                    cleaning_pipeline_hash="", notes="no cleaning pipeline hash"))
    # 4. PII present
    bad.append(_doc("doc-bad-pii-000", "src-forum-2", "general_web", "train",
                    [_seg("text", "Contact me at test.person@example.com or +91-90000-11111. "
                                  + " ".join(rng.choice(WEB_TAIL) for _ in range(3)), True)],
                    "en", "Latn", pii_status="detected", notes="unredacted PII"))
    # 5. language not validated
    bad.append(_doc("doc-bad-langid-000", "src-mixed-3", "indic", "train",
                    [_seg("text", "मिश्रित text with unverified language identification.", True)],
                    "hi", "Deva", lang_validated=False, notes="language id below threshold"))
    # 6. contaminated with a verbatim evaluation span
    contaminated = ("Study notes. " + test[2]["text"][:180] +
                    " Additional commentary follows for realism.")
    bad.append(_doc("doc-bad-contaminated-000", "src-scraped-4", "general_web", "train",
                    [_seg("text", contaminated, True)], "en", "Latn",
                    notes="verbatim overlap with TDES-BENCH-v1/item2"))
    # 7. an evaluation document that someone tried to submit as training data
    smuggled = dict(test[0])
    bad.append(_doc("doc-bad-evalsmuggle-000", "bench-tdes", "general_web", "train",
                    [_seg("text", smuggled["text"], True)], "en", "Latn",
                    notes="test-set document relabelled as training data"))

    return {"train": train, "validation": validation, "test": test, "rejected_seed": bad}


def write_corpus(corpus: Dict[str, List[dict]]) -> str:
    path = os.path.join(PATHS["corpus"], "corpus.json")
    payload = {k: [{kk: vv for kk, vv in d.items()} for d in v] for k, v in corpus.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return path
