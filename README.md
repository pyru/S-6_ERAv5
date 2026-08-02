# TDES — Training Data Execution System (V5, Session 6)

A small but complete data system that proves what a training run consumed, why
it consumed it, what the model learned from it, and how the run can be
reconstructed.

```
documents → tokenized shards → manifests → mixture schedule → packing → batches
→ training → consumption ledger → learning ledger → checkpoint → crash
→ resume → replay → fork → audit
```

The corpus, tokenizer and model are deliberately tiny. Nothing about the
*claims* is tiny: every number in the evidence bundle is re-derived from files
on disk, the crash is a real process death, and replay reconstructs historical
batches without reading the original ones.

---

## Run it

```bash
python run_demo.py
```

Requires Python 3.9+ and NumPy. No network, no dataset downloads, no other
dependencies. Runs in about 50 seconds and regenerates
`submission_artifacts/` from scratch each time.

```bash
pip install -r requirements.txt      # numpy only
python run_demo.py
python -m unittest discover tests    # tests can also run standalone
```

Exit code is 0 only when all 14 evidence requirements pass **and** the full
invariant suite passes.

---

## What the demo actually does

| Phase | Process | What it proves |
| --- | --- | --- |
| `prepare` | corpus → tokenizer → shards → registry → mixture → packs | frozen tokenizer, sealed shards, admission gate, eval firewall, compiled curriculum, packing policies |
| `train --crash-at 27` | 48-step run, checkpoint every 8 | consumption + learning ledgers, OPUS trail, checkpoints bound to ledger offsets; then `os._exit(70)` mid-run |
| `resume` | restarts from `ckpt-main-0024` | the next batch served is byte-identical to the batch the dead process had already recorded for step 25 |
| `replay` | independent rebuild of steps 1–16 | batch ids, batch hashes, token hashes, loss-mask hashes and token spans for steps 9–16 match the original run |
| `fork` | new branch from `ckpt-main-0016` | same model state, deliberately different data stream, divergence recorded |
| `audit` | reads only files | chain integrity, no gaps/duplicates, mixture compliance, floors, firewall, learning trace, "which shards trained steps 24–32?" |
| `perf` | reads only ledgers | throughput and packing numbers, each reconstructible from raw counters |
| `evidence` | reads only artifacts | `evidence.json` + `evidence.md` |

Each phase is a separate OS process (`python -m tdes.worker <cmd>`), so the
crash cannot be faked with an exception handler and recovery has to come from
durable state: checkpoints plus hash-chained ledgers.

---

## Architecture

```
tdes/
  hashing.py     canonical JSON, sha256, hash chains, merkle roots, stable RNG
  config.py      the frozen configuration; stream identity is a function of it
  corpus.py      deterministic synthetic corpus, 6 lanes + eval/validation
                 + 7 deliberately-bad documents for the admission gate
  tokenizer.py   frozen byte-level BPE, Indic-safe normalization
  shards.py      admission gate, immutable sealed shards, manifests, ShardStore
  registry.py    evaluation/validation registry + the two firewall enforcement points
  mixture.py     curriculum stages → per-step integer lane quotas
  packing.py     six packing policies, loss/attention masks, position ids, verifier
  opus.py        candidate scoring and accept / reject / defer / floor-override
  dataloader.py  the training stream: quotas → candidates → OPUS → packed batch
  model.py       tiny float64 causal-attention transformer with hand-written backprop
  ledger.py      append-only hash-chained ledgers + supersede semantics
  checkpoint.py  model + optimizer + stream state + ledger offsets and heads
  trainer.py     the training loop and its ledger writes
  worker.py      process entry points: prepare/train/resume/replay/fork/audit/perf/evidence
  audit.py       re-derives every invariant from disk
  perf.py        throughput and packing efficiency, with raw counters
  evidence.py    the evidence bundle
tests/
  test_units.py      invariants that need no run (tokenizer, packing, model, mixture, hashing)
  test_artifacts.py  re-verification of the generated run
run_demo.py      one command: orchestrates every phase, runs the tests twice
```

---

## Design decisions

### 1. Batch identity is a pure function of four things on disk

`(branch config, mixture schedule, immutable pack pools, frozen proxy
checkpoint)` fully determine the batch at any step. That is why replay is
meaningful: `tdes.worker replay` never reads the original batch arrays — it
rebuilds the stream from scratch (no trainer, no gradients) and only then
compares hashes against the ledger.

OPUS scores against a **frozen proxy checkpoint**, rebound only at checkpoint
boundaries (`scoring_checkpoint_id` in every decision record). Selection
therefore depends on model state — as it should — while remaining exactly
reproducible, because the proxy is a file.

### 2. The ledger is append-only; the *effective* stream is what matters

Crash recovery does not truncate the ledger. The checkpoint stores each
ledger's `(offset, head_hash)`. On resume the chain is verified up to that
offset and a `ledger_rollback` record is appended that marks the orphaned
records as superseded. The effective consumption stream is then read as
"records not superseded", and the invariant is exact: **one `batch_served`
record per step, 4 `microbatch_consumed` records per step, contiguous 1..48,
no duplicates.**

The 3 steps lost to the crash are not swept under the rug — they are counted
in `performance.json` as `crash_wasted_steps` / `crash_wasted_positions`.

### 3. Attention isolation is per *sample*, loss masking is per *role*

A packed window carries two different groupings:

* `segment_ids` — the attention isolation unit. Two samples packed into one
  window can never attend to each other (causal **and** block-diagonal).
* `role_ids` — prompt / assistant / tool-call / tool-observation / thought /
  answer, which decides what bears loss.

If attention were isolated per role, an SFT response could not see its own
prompt. If loss were masked per segment, the prompt→response transition — the
single most important position in an SFT sample — would be dropped. The loss
contract is stated once and checked everywhere:

```
loss_mask[t] == 1  iff  segment_ids[t+1] == segment_ids[t] != 0
                        and the target token t+1 has a loss-bearing role
```

`verify_pack()` re-derives this from the stored `target_loss`, so the check is
not circular. It runs at pack-build time, at serve time inside the trainer, in
the packing report for every policy × lane combination, and again in the tests.

The model is small but genuinely consumes all three signals: a test mutates a
token in segment 2 and asserts segment 1's logits are bit-identical; another
changes position ids and asserts the logits *do* change; another asserts an
all-zero loss mask produces an exactly-zero gradient. Hand-written backprop is
checked against central finite differences.

### 4. Packing policy is a per-lane decision, and the cost is reported

`packing_report.json` runs all six policies over all six lanes and reports
utilization, useful-token ratio, padding, boundary crossings, split samples and
`structure_safe`. The training stream then uses one policy per lane
(`concat_chop` for web/Indic, `best_fit` for code, `greedy` for maths,
`structure_preserving` for agentic and reasoning). Agentic packs land at ~58%
utilization — that is the real price of never splitting a trajectory, and it is
in the report rather than hidden.

### 5. Two firewall enforcement points, not one

1. **Admission** — a candidate document that overlaps registered evaluation
   data (sliding 48-char window fingerprints, plus canary strings, plus exact
   text hashes) can never become a shard. The corpus deliberately contains a
   contaminated document and a smuggled benchmark item; both are blocked with
   reasons in `admission_report.json`.
2. **Serving** — every packed batch has its *loss-bearing* tokens scanned
   against 13-gram token fingerprints before it can reach the optimizer, and a
   blocked batch raises rather than trains.

Validation data is registered as `eval_read_only`: it is forward-passed at every
checkpoint, every read is written to an access log with
`gradient_bearing: false`, and the audit asserts that no held-out `doc_id`
appears anywhere in the consumption ledger.

### 6. Protected floors are enforced twice

At compile time, `enforce_floors()` raises a starved lane's weight to its floor
and takes the deficit from the unfloored lanes. At serve time, the stream
computes each lane's actual cumulative share; when a floored lane falls under
its floor, OPUS rescues candidates it would otherwise have benched and records
`protected_floor_override: true`. Both mechanisms show up in the audit
(`mixture_compliance.json` and the OPUS status counts).

### 7. Provenance is verified during training, not just asserted

Every served sequence is re-read from its sealed shard through the loader cache
and its token span re-hashed against the manifest. This makes the token spans
in the consumption ledger provably real, and it is also what produces the
genuine cache hit rate and shard read latency in `performance.json`.

### 8. Throughput numbers ship with the counters that produced them

`performance.json` carries `raw_counters` (from the trainer) and
`ledger_recomputation` (re-derived from the ledgers), asserts they agree, and
states the formula. Packing utilization in the report is checked against
`performed_real_tokens / performed_positions` by a test. A number that cannot
be reconstructed does not belong in the report.

### 9. The evidence bundle is a verifier, not a template

`tdes/evidence.py` opens the generated manifests, ledgers, checkpoints and
reports and evaluates predicates against them. Each requirement's PASS/FAIL
falls out of its checks; each check records what was observed and what was
expected. A test asserts that `result` really does follow from `checks`, that
every referenced artifact exists, and that `evidence_hash` covers the document.

---

## Generated artifacts

```
submission_artifacts/
  run.log                      complete execution log with [PASS]/[FAIL] events
  run_events.jsonl             structured mirror of run.log
  evidence.json                machine-readable evidence bundle
  evidence.md                  human-readable summary
  performance.json             throughput, packing, cache, OPUS rates + raw counters
  manifests/
    tokenizer.json             frozen tokenizer spec + tokenizer_hash
    shards/*.json              one immutable manifest per shard
    manifest_index.json        shard index + merkle root
    eval_registry.json         benchmark/validation registry, fingerprints, access log
    admission_report.json      every admitted and rejected candidate, with reasons
    mixture_schedule_main.json per-step lane quotas, floors, reserves, scarcity
    mixture_schedule_fork-a.json
    packing_report.json        6 policies × 6 lanes
    prepare_summary.json
  ledgers/<branch>/
    consumption.jsonl          batch_served + microbatch_consumed (hash-chained)
    opus.jsonl                 every candidate decision
    learning.jsonl             per-sequence loss before/after, grad norm, classification
    token_trace.jsonl          per-token loss/perplexity linked to doc + shard
    firewall.jsonl             per-batch eval-overlap scan
    control.jsonl              checkpoints, rollbacks, validation reads, forks, perf
  checkpoints/<branch>/*.json  params + optimizer + stream state + ledger offsets/heads
  shards/*.bin                 read-only tokenized shards (+ .side.json role/loss arrays)
  packs/*.jsonl                immutable packed windows per lane
  reports/
    audit.json  resume_report.json  replay_report.json  fork_report.json
    learning_report.json  mixture_compliance.json  test_results*.json
    phase_timings.json
```

### Key log markers

```
[PASS] tokenizer_hash_verified
[PASS] eval_shard_blocked
[PASS] checkpoint_saved
[PASS] resume_next_batch_matched
[PASS] replay_hash_matched
[PASS] no_skipped_or_repeated_batches
[PASS] fork_stream_diverged
[PASS] audit_completed
[PASS] performance_measured
```

---

## Tests

77 tests across two files, run twice by `run_demo.py` (once before the evidence
bundle so its result can feed the bundle, once after so the evidence checks
themselves are live).

`test_units.py` — tokenizer determinism / lossless roundtrip / ZWJ–ZWNJ
preservation / no NFKC or casefold; all six packing policies against the
invariant verifier; pad, position-reset, boundary and attention-isolation
rules; corruption detection; analytic vs finite-difference gradients; attention
mask isolation; position-id sensitivity; loss-mask gradient control; mixture
allocation purity and exactness; floor enforcement; anneal reserve
withhold/release; warmup blending; hash-chain and merkle tamper detection;
contamination and smuggled-eval detection.

`test_artifacts.py` — shard verification, read-only enforcement, content-hash
mutation detection, merkle match, span reconstruction; pack pools rebuilt from
shards reproduce every `pack_hash`; ledger chains and tamper detection;
one-batch-per-step with no gaps or duplicates; superseded records exist;
microbatch fan-out; token spans point at real shard content; every consumed
pack has a learning event; firewall coverage; OPUS trail completeness; the
checkpoint↔ledger-offset binding; resume/replay/fork reports; performance
reconstructibility; and that the evidence bundle agrees with the artifacts.

---

## Scope and honesty notes

* Multi-rank execution is **simulated** in one process: sequences are assigned
  to `rank`/`accum` slots by a fixed topology (2 ranks × 2 grad-accum steps × 2
  micro-batch) and each microbatch gets its own ledger record. No real
  multi-process dataloader is claimed.
* Shard read latency and cache hit rate are measured from real file reads
  through `ShardStore`, but on a local disk with tiny files — the numbers are
  reconstructible, not representative of cluster storage.
* The corpus is synthetic and generated from a seed. That is what makes the run
  byte-reproducible without shipping data.
* `throughput` is reported over the work actually performed (including the
  three steps lost to the crash); the committed-stream figure is reported
  separately so neither number is quietly flattering.
