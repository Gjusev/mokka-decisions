"""Backend for our own trained OptionScorer checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import DecisionCase, Option
from ..model import OptionScorer, ScorerInference, load_checkpoint
from .base import Backend


class CustomBackend(Backend):
    name = "mokka_option_scorer"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        encoder_name: str = "jhu-clsp/mmBERT-base",
        device: str | None = None,
        temperature: float = 1.0,
        batch_decisions: int = 16,
        max_length: int = 256,
    ):
        import torch
        from transformers import AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        model = OptionScorer(encoder_name)
        load_checkpoint(model, checkpoint)
        extra = Path(str(checkpoint)).with_suffix(".json")
        revision = ""
        if extra.exists():
            revision = json.loads(extra.read_text(encoding="utf-8")).get("config_hash", "")
        self.scorer = ScorerInference(
            model,
            self.tokenizer,
            device=device,
            temperature=temperature,
            batch_decisions=batch_decisions,
            max_length=max_length,
        )
        self.model_revision = f"option-scorer@{(revision or 'unversioned')[:12]}"

    def score_one(self, case: DecisionCase) -> dict[str, float]:
        # single-case path keeps backend timing per case (batch=1 latency);
        # batched throughput is measured by run_experiment separately.
        return self.scorer.score([case])[0]

    def warmup(self) -> None:
        dummy = DecisionCase(
            id="warmup",
            group_id="warmup",
            source="warmup",
            language="en",
            domain="banking",
            state="warmup",
            question="warmup?",
            options=(Option(id="a", description="x"), Option(id="b", description="y")),
        )
        self.scorer.score([dummy])
