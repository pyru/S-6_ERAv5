"""Global, frozen configuration for the demonstration run.

Everything that influences the byte-identity of an artifact lives here so the
whole run is a pure function of this file plus the code version.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List

# ---------------------------------------------------------------- versions --
DATALOADER_VERSION = "tdes-loader-1.0.0"
PACKER_VERSION = "tdes-packer-1.0.0"
CLEANING_PIPELINE_VERSION = "clean-v5.2"
OPUS_PROXY_VERSION = "opus-proxy-1.0.0"
CODE_VERSION = "tdes-1.0.0"

# ------------------------------------------------------------------- seeds --
MASTER_SEED = 20260802
CORPUS_SEED = 1337
TOKENIZER_SEED = 4242
INIT_SEED = 909

# --------------------------------------------------------------- tokenizer --
VOCAB_SIZE = 512

# ----------------------------------------------------------- batch geometry --
SEQ_LEN = 256
LONG_SEQ_LEN = 512          # long-context lane window
WORLD_SIZE = 2              # simulated ranks (GPUs)
MICRO_BATCH = 2             # sequences per rank per micro-step
GRAD_ACCUM = 2              # micro-steps per optimizer step
SEQS_PER_STEP = WORLD_SIZE * MICRO_BATCH * GRAD_ACCUM      # = 8
TOKENS_PER_STEP = SEQS_PER_STEP * SEQ_LEN                  # = 1024

# ------------------------------------------------------------ run schedule --
TOTAL_STEPS = 48
CHECKPOINT_EVERY = 8
CRASH_AT_STEP = 27          # crash *before* consuming this step
REPLAY_FROM, REPLAY_TO = 9, 16
FORK_FROM_STEP = 16
FORK_STEPS = 8

# ---------------------------------------------------------------- training --
D_MODEL = 32
D_FF = 64
LEARNING_RATE = 0.6
MOMENTUM = 0.9
GRAD_CLIP = 1.5

# ------------------------------------------------------------------- lanes --
LANES: List[str] = [
    "general_web",
    "code",
    "math_science",
    "indic",
    "agentic",
    "reasoning",
]

# lane -> packing policy used by the live training stream
LANE_PACKING_POLICY: Dict[str, str] = {
    "general_web": "concat_chop",
    "code": "best_fit",
    "math_science": "greedy",
    "indic": "concat_chop",
    "agentic": "structure_preserving",
    "reasoning": "structure_preserving",
}

ALLOWED_LICENSE_TIERS = {"public_domain", "permissive", "cc_by"}
BLOCKED_LICENSE_TIERS = {"noncommercial", "unknown", "proprietary"}

# --------------------------------------------------------------- OPUS gate --
OPUS_ACCEPT_THRESHOLD = 0.55
OPUS_DEFER_THRESHOLD = 0.50
OPUS_OVERGENERATION = 1.75   # candidates generated per required sequence

# --------------------------------------------------------------- artifacts --
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "submission_artifacts")

PATHS = {
    "art": ART,
    "run_log": os.path.join(ART, "run.log"),
    "evidence_json": os.path.join(ART, "evidence.json"),
    "evidence_md": os.path.join(ART, "evidence.md"),
    "performance": os.path.join(ART, "performance.json"),
    "manifests": os.path.join(ART, "manifests"),
    "shard_manifests": os.path.join(ART, "manifests", "shards"),
    "ledgers": os.path.join(ART, "ledgers"),
    "checkpoints": os.path.join(ART, "checkpoints"),
    "shards": os.path.join(ART, "shards"),
    "packs": os.path.join(ART, "packs"),
    "reports": os.path.join(ART, "reports"),
    "corpus": os.path.join(ART, "corpus"),
}


def ensure_dirs() -> None:
    for key in ("art", "manifests", "shard_manifests", "ledgers", "checkpoints",
                "shards", "packs", "reports", "corpus"):
        os.makedirs(PATHS[key], exist_ok=True)


@dataclass(frozen=True)
class BranchConfig:
    """Defines a data branch: identity of the stream, not of the model."""
    branch_id: str
    seed: int
    parent_branch: str = ""
    fork_step: int = 0
    mixture_override: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


MAIN_BRANCH = BranchConfig(branch_id="main", seed=MASTER_SEED)
FORK_BRANCH = BranchConfig(
    branch_id="fork-a",
    seed=MASTER_SEED + 7,
    parent_branch="main",
    fork_step=FORK_FROM_STEP,
    # the fork deliberately changes the stream: more reasoning, less web
    mixture_override={"reasoning": 0.22, "general_web": 0.20},
)


def config_fingerprint() -> dict:
    """The subset of config that defines stream identity."""
    return {
        "code_version": CODE_VERSION,
        "dataloader_version": DATALOADER_VERSION,
        "packer_version": PACKER_VERSION,
        "opus_proxy_version": OPUS_PROXY_VERSION,
        "master_seed": MASTER_SEED,
        "seq_len": SEQ_LEN,
        "seqs_per_step": SEQS_PER_STEP,
        "world_size": WORLD_SIZE,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "vocab_size": VOCAB_SIZE,
        "lanes": LANES,
        "lane_packing_policy": LANE_PACKING_POLICY,
        "opus": {
            "accept": OPUS_ACCEPT_THRESHOLD,
            "defer": OPUS_DEFER_THRESHOLD,
            "overgeneration": OPUS_OVERGENERATION,
        },
    }
