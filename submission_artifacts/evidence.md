# TDES Evidence Summary

**Overall result: PASS** (14/14 requirements)

- tokenizer hash: `95d42d29ec24d6b2b3f6430e232bbc03d1714f904a63598d2e32caa555758e05`
- shard merkle root: `ba8bd10fdafd7da5b91a9bbb5d15be886eabd44d6fbf33a142b1d2864d6f165f`
- mixture schedule hash: `bed043ecd7ef5078c6fd6baf58e56517c2f8b8057b89b23c500dfca956a25c56`
- steps: 48, crash at step 27, replay interval [9, 16]
- evidence hash: `e0d3dc7f7c9d5ff886c5e03ebe168905cbb7218252435a40d1910e3d1b874056`

## Required summary

| Requirement | Result | Evidence |
| --- | --- | --- |
| Tokenizer integrity | **PASS** | Manifest record - `manifests/tokenizer.json`; `manifests/manifest_index.json` |
| Evaluation and validation firewall | **PASS** | Blocked-shard event - `manifests/eval_registry.json`; `ledgers/main/firewall.jsonl` |
| Packing, masks and batch correctness | **PASS** | Packed-batch report - `manifests/packing_report.json`; `packs` |
| Mixture schedule, floors and curriculum | **PASS** | Planned versus actual shares - `manifests/mixture_schedule_main.json`; `reports/mixture_compliance.json` |
| OPUS acceptance, rejection, deferral, override | **PASS** | Candidate decision records - `ledgers/main/opus.jsonl`; `reports/audit.json` |
| Crash recovery: no skipped or repeated batches | **PASS** | Expected and resumed batch ids - `reports/resume_report.json`; `ledgers/main/control.jsonl` |
| Replay of the historical data stream | **PASS** | Original and replay hashes - `reports/replay_report.json` |
| Learning ledger and token-level loss trace | **PASS** | Loss linked to source data - `ledgers/main/learning.jsonl`; `ledgers/main/token_trace.jsonl` |
| Throughput and packing efficiency | **PASS** | Performance report - `performance.json` |

## All requirements

| # | Requirement | Result | Checks | Evidence |
| --- | --- | --- | --- | --- |
| 1 | End-to-end execution | **PASS** | 2/2 | `run.log`; `run_events.jsonl` |
| 2 | Tokenizer integrity | **PASS** | 5/5 | `manifests/tokenizer.json`; `manifests/manifest_index.json` |
| 3 | Immutable shards and manifests | **PASS** | 4/4 | `manifests/shards`; `manifests/manifest_index.json`; `manifests/admission_report.json` |
| 4 | Packing, masks and batch correctness | **PASS** | 5/5 | `manifests/packing_report.json`; `packs` |
| 5 | Mixture schedule, floors and curriculum | **PASS** | 3/3 | `manifests/mixture_schedule_main.json`; `reports/mixture_compliance.json` |
| 6 | OPUS acceptance, rejection, deferral, override | **PASS** | 7/7 | `ledgers/main/opus.jsonl`; `reports/audit.json` |
| 7 | Training consumption ledger | **PASS** | 4/4 | `ledgers/main/consumption.jsonl`; `reports/audit.json` |
| 8 | Learning ledger and token-level loss trace | **PASS** | 5/5 | `ledgers/main/learning.jsonl`; `ledgers/main/token_trace.jsonl`; `reports/learning_report.json` |
| 9 | Crash recovery: no skipped or repeated batches | **PASS** | 6/6 | `reports/resume_report.json`; `ledgers/main/control.jsonl`; `checkpoints/main` |
| 10 | Replay of the historical data stream | **PASS** | 6/6 | `reports/replay_report.json` |
| 11 | Fork from an earlier checkpoint | **PASS** | 4/4 | `reports/fork_report.json`; `ledgers/fork-a/control.jsonl` |
| 12 | Evaluation and validation firewall | **PASS** | 5/5 | `manifests/eval_registry.json`; `ledgers/main/firewall.jsonl`; `manifests/admission_report.json` |
| 13 | Throughput and packing efficiency | **PASS** | 4/4 | `performance.json` |
| 14 | Automated invariant tests | **PASS** | 2/2 | `reports/test_results.json` |

## Check detail

### End-to-end execution - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `all_phases_logged` | PASS | ["audit_completed", "batch_packed", "branch_forked", "checkpoint_saved", "crash_simulated"... | ["shard_created", "manifests_validated", "eval_shard_blocked", "mixture_compiled", "batch_... |
| `no_FAIL_events` | PASS | [] | [] |

- evidence: `run.log` -> `whole file` (complete execution log)
- evidence: `run_events.jsonl` -> `structured mirror of run.log`

### Tokenizer integrity - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `tokenizer_hash_verified` | PASS | - | - |
| `roundtrip_lossless` | PASS | - | - |
| `indic_zero_width_preserved` | PASS | - | - |
| `all_shards_carry_frozen_hash` | PASS | ["95d42d29ec24d6b2b3f6430e232bbc03d1714f904a63598d2e32caa555758e05"] | ["95d42d29ec24d6b2b3f6430e232bbc03d1714f904a63598d2e32caa555758e05"] |
| `normalization_is_indic_safe` | PASS | {"casefold": false, "crlf_folded": true, "form": "NFC", "nfkc": false, "strip_zero_width":... | - |

- evidence: `manifests/tokenizer.json` -> `tokenizer_hash`
- evidence: `manifests/manifest_index.json` -> `tokenizer_hash`

### Immutable shards and manifests - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `all_shards_verify` | PASS | [] | - |
| `merkle_root_match` | PASS | ba8bd10fdafd7da5b91a9bbb5d15be886eabd44d6fbf33a142b1d2864d6f165f | ba8bd10fdafd7da5b91a9bbb5d15be886eabd44d6fbf33a142b1d2864d6f165f |
| `shard_count_matches_index` | PASS | 27 | 27 |
| `admission_gate_rejected_bad_docs` | PASS | {"duplicate_document": 1, "eval_contamination": 2, "language_not_validated": 1, "license_t... | - |

- evidence: `manifests/shards` -> `one manifest per shard`
- evidence: `manifests/manifest_index.json` -> `merkle_root`
- evidence: `manifests/admission_report.json` -> `rejection_reason_counts`

### Packing, masks and batch correctness - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `zero_invariant_violations_across_all_policies` | PASS | 0 | 0 |
| `policies_compared` | PASS | ["best_fit", "concat_chop", "greedy", "long_context", "pad_only", "structure_preserving"] | - |
| `structured_lanes_use_structure_safe_policy` | PASS | ["structure_preserving", "best_fit", "concat_chop", "concat_chop", "greedy", "structure_pr... | - |
| `serve_time_invariants_enforced` | PASS | - | - |
| `no_batch_invariant_failures` | PASS | - | - |

- evidence: `manifests/packing_report.json` -> `rows[].invariant_violations`
- evidence: `packs` -> `packed windows with masks/positions/segments`

### Mixture schedule, floors and curriculum - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `planned_vs_actual_within_tolerance` | PASS | 0.0 | <= 0.05 |
| `protected_floors_respected` | PASS | [] | [] |
| `stages_compiled` | PASS | - | - |

- evidence: `manifests/mixture_schedule_main.json` -> `per_step[].alloc_sequences`
- evidence: `reports/mixture_compliance.json` -> `rows[]`

### OPUS acceptance, rejection, deferral, override - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `every_consumed_batch_has_a_decision` | PASS | [] | [] |
| `selection_matches_consumption` | PASS | - | - |
| `acceptances_recorded` | PASS | {"accepted": 265, "deferred": 82, "deferred_promoted": 89, "protected_floor_override": 9, ... | - |
| `rejections_retained` | PASS | 320 | - |
| `deferrals_recorded` | PASS | {"accepted": 265, "deferred": 82, "deferred_promoted": 89, "protected_floor_override": 9, ... | - |
| `protected_floor_override_exercised` | PASS | 9 | - |
| `scored_against_frozen_proxy_checkpoints` | PASS | ["ckpt-main-0000", "ckpt-main-0008", "ckpt-main-0016", "ckpt-main-0024", "ckpt-main-0032",... | - |

- evidence: `ledgers/main/opus.jsonl` -> `payload.status / rejection_reason`
- evidence: `reports/audit.json` -> `opus`

### Training consumption ledger - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `hash_chains_verify` | PASS | {"consumption": {"count": 255, "head": "bf5056e39dea785ee8b34743852b165bb180d842a88fcabd4e... | - |
| `one_batch_record_per_step` | PASS | [1, 48] | [1, 48] |
| `microbatch_fanout_correct` | PASS | 4 | - |
| `checkpoint_range_query_reconstructs_data` | PASS | 136 | - |

- evidence: `ledgers/main/consumption.jsonl` -> `payload.token_spans / loss_mask_hash`
- evidence: `reports/audit.json` -> `ledgers_main`

### Learning ledger and token-level loss trace - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `learning_events_cover_every_consumed_pack` | PASS | - | - |
| `token_level_trace_linked_to_source_data` | PASS | 1129 | - |
| `loss_decreased_over_the_run` | PASS | [6.15946, 4.253423] | - |
| `per_shard_report_cards_generated` | PASS | 27 | - |
| `repeated_pass_effect_measured` | PASS | [{"exposures": 47, "mean_loss_delta": 0.0836324, "repeated_pass_number": 0}, {"exposures":... | - |

- evidence: `ledgers/main/learning.jsonl` -> `payload.loss_delta / shard_ids`
- evidence: `ledgers/main/token_trace.jsonl` -> `payload.ppl per token`
- evidence: `reports/learning_report.json` -> `shard_report_cards`

### Crash recovery: no skipped or repeated batches - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `crash_was_a_real_process_exit` | PASS | [{"exit_code": 70, "last_checkpoint": "ckpt-main-0024", "step": 27, "uncheckpointed_steps"... | - |
| `resume_next_batch_matched` | PASS | batch-main-00025-73a43311fa7e | batch-main-00025-73a43311fa7e |
| `expected_batch_came_from_the_pre_crash_ledger` | PASS | [25, 26, 27] | - |
| `effective_stream_contiguous` | PASS | 48 | 48 |
| `no_duplicate_steps` | PASS | [] | [] |
| `ledger_chain_valid_at_checkpoint` | PASS | - | - |

- evidence: `reports/resume_report.json` -> `expected_batch vs resumed_batch`
- evidence: `ledgers/main/control.jsonl` -> `type=ledger_rollback`
- evidence: `checkpoints/main` -> `ledger_offsets + ledger_heads`

### Replay of the historical data stream - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `replay_all_hashes_match` | PASS | [] | [] |
| `batch_ids_match` | PASS | - | - |
| `token_spans_match` | PASS | - | - |
| `loss_mask_hashes_match` | PASS | - | - |
| `interval_covered` | PASS | [9, 16] | [9, 16] |
| `digests_equal` | PASS | c223baa0ed23e80abfbd775fc7eded407e68195b8eeb9889c8b984d682f27e81 | c223baa0ed23e80abfbd775fc7eded407e68195b8eeb9889c8b984d682f27e81 |

- evidence: `reports/replay_report.json` -> `rows[]`

### Fork from an earlier checkpoint - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `fork_recorded` | PASS | - | - |
| `data_stream_diverged` | PASS | [] | [] |
| `model_state_inherited_from_parent` | PASS | ckpt-main-0016 | - |
| `fork_ledger_chain_ok` | PASS | - | - |

- evidence: `reports/fork_report.json` -> `rows[].fork_batch_id vs main`
- evidence: `ledgers/fork-a/control.jsonl` -> `type=branch_forked`

### Evaluation and validation firewall - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `eval_shard_blocked_at_admission` | PASS | ["doc-bad-contaminated-000", "doc-bad-evalsmuggle-000"] | - |
| `zero_eval_overlap_in_served_batches` | PASS | 0 | 0 |
| `no_eval_docs_in_training_stream` | PASS | [] | [] |
| `validation_never_gradient_bearing` | PASS | 6 | - |
| `every_batch_scanned` | PASS | 51 | >= 48 |

- evidence: `manifests/eval_registry.json` -> `entries[].never_train`
- evidence: `ledgers/main/firewall.jsonl` -> `eval_overlap_hits per batch`
- evidence: `manifests/admission_report.json` -> `blocked_eval_docs`

### Throughput and packing efficiency - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `useful_tokens_per_sec_reported` | PASS | {"accepted_tokens_per_sec_after_opus": 8839.29, "committed_useful_tokens_per_sec": 7841.63... | - |
| `counters_reconstructible_from_ledger` | PASS | [120832, 120832] | - |
| `packing_utilization_reported` | PASS | 0.910851 | - |
| `packing_utilization_matches_ledger` | PASS | - | - |

- evidence: `performance.json` -> `throughput / packing / raw_counters`

### Automated invariant tests - PASS

| Check | Result | Observed | Expected |
| --- | --- | --- | --- |
| `test_suite_executed` | PASS | 77 | - |
| `all_tests_passed` | PASS | {"errors": 0, "failures": 0, "total": 77} | - |

- evidence: `reports/test_results.json` -> `unittest summary`

## Artifact inventory

128 files, 12867482 bytes

| Directory | Files |
| --- | --- |
| `.` | 3 |
| `checkpoints` | 8 |
| `corpus` | 1 |
| `ledgers` | 12 |
| `manifests` | 35 |
| `packs` | 7 |
| `reports` | 8 |
| `shards` | 54 |
