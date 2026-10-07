"""The own decision head: a shared-weight option scorer over an open encoder.

Architecture (the part we train):

    encoder("state", "question + option description") -> pooled vector per option
    head(pooled) -> scalar score per option
    softmax over the valid options of each decision (padding masked to -inf)

Every option of a decision is scored with the *same* encoder weights and the
same head — no per-option parameters, no label-id embedding. This is what lets
the model score option sets it never saw in training. Cost: one encoder pass
per option (a K-option decision costs K passes); measured in evaluation, not
assumed to be shared.

The encoder itself is pretrained by third parties (see model card); we fine-tune
it jointly with the new head.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import DecisionCase

DEFAULT_ENCODER = "jhu-clsp/mmBERT-base"


class OptionScorer(nn.Module):
    def __init__(self, encoder_name: str = DEFAULT_ENCODER, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        from transformers import AutoModel  # lazy: keeps contract imports cheap

        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,          # (total_seqs, L)
        attention_mask: torch.Tensor,     # (total_seqs, L)
        token_type_ids: torch.Tensor | None,
        decision_index: torch.Tensor,     # (total_seqs,) which decision each seq belongs to
        num_decisions: int,
        max_options: int,
        option_slot: torch.Tensor,        # (total_seqs,) which option slot each seq fills
    ) -> torch.Tensor:
        """Return logits (num_decisions, max_options), -inf at padded slots."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # CLS token as the option representation: deterministic across
        # transformers versions (pooler_output availability varies in v5)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        scores = self.head(pooled).squeeze(-1)  # (total_seqs,)

        logits = torch.full(
            (num_decisions, max_options), -1e9, dtype=scores.dtype, device=scores.device
        )
        logits[decision_index, option_slot] = scores
        return logits

    def parameter_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups


def encode_cases(
    tokenizer,
    cases: Sequence[DecisionCase],
    *,
    max_length: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Flatten decisions x options into tokenized sequences + index tensors."""
    pairs_a: list[str] = []
    pairs_b: list[str] = []
    decision_index: list[int] = []
    option_slot: list[int] = []
    for di, case in enumerate(cases):
        for si, option in enumerate(case.options):
            pairs_a.append(case.state)
            pairs_b.append(f"{case.question} Option: {option.description}")
            decision_index.append(di)
            option_slot.append(si)

    enc = tokenizer(
        pairs_a,
        pairs_b,
        padding=True,
        truncation="longest_first",
        max_length=max_length,
        return_tensors="pt",
    )
    num_decisions = len(cases)
    max_options = max(len(c.options) for c in cases)
    batch = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "decision_index": torch.tensor(decision_index, dtype=torch.long, device=device),
        "option_slot": torch.tensor(option_slot, dtype=torch.long, device=device),
        "num_decisions": num_decisions,
        "max_options": max_options,
    }
    tt = enc.get("token_type_ids")
    batch["token_type_ids"] = tt.to(device) if tt is not None else None
    return batch


def case_targets(cases: Sequence[DecisionCase], device="cpu") -> torch.Tensor:
    """Option *position* of the gold option for each decision."""
    targets = []
    for case in cases:
        targets.append(case.option_ids.index(case.target_option))
    return torch.tensor(targets, dtype=torch.long, device=device)


def save_checkpoint(
    model: OptionScorer,
    path: str | Path,
    *,
    extra: Mapping | None = None,
) -> None:
    """Atomic save: weights as safetensors, optional sidecar metadata as JSON."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(model.state_dict()), str(tmp), metadata={"format": "pt"})
    if extra is not None:
        meta_path = path.with_suffix(".json")
        meta_tmp = meta_path.with_suffix(".json.tmp")
        meta_tmp.write_text(json.dumps(extra, indent=2, sort_keys=True), encoding="utf-8")
        meta_tmp.replace(meta_path)
    tmp.replace(path)


def load_checkpoint(model: OptionScorer, path: str | Path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"missing keys in checkpoint: {missing[:5]}")


class ScorerInference:
    """Batched scorer for evaluation/inference; returns option-id probabilities."""

    def __init__(
        self,
        model: OptionScorer,
        tokenizer,
        *,
        device: str = "cpu",
        temperature: float = 1.0,
        max_length: int = 256,
        batch_decisions: int = 16,
    ):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        self.batch_decisions = batch_decisions
        self.model_revision = ""  # set by loaders that know the checkpoint

    @torch.no_grad()
    def score(self, cases: Sequence[DecisionCase]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for start in range(0, len(cases), self.batch_decisions):
            chunk = cases[start : start + self.batch_decisions]
            batch = encode_cases(
                self.tokenizer, chunk, max_length=self.max_length, device=self.device
            )
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                decision_index=batch["decision_index"],
                num_decisions=batch["num_decisions"],
                max_options=batch["max_options"],
                option_slot=batch["option_slot"],
            )
            if self.temperature != 1.0:
                logits = logits / self.temperature
            for case, row in zip(chunk, logits):
                ids = case.option_ids
                valid = [row[i].item() for i in range(len(ids))]
                probs = torch.softmax(torch.tensor(valid), dim=-1)
                out.append({oid: float(p) for oid, p in zip(ids, probs)})
        return out

    @torch.no_grad()
    def score_logits(self, cases: Sequence[DecisionCase]) -> list[dict[str, float]]:
        """Raw logits (no temperature, pre-softmax) per option id, for calibration."""
        probs = []
        for start in range(0, len(cases), self.batch_decisions):
            chunk = cases[start : start + self.batch_decisions]
            batch = encode_cases(
                self.tokenizer, chunk, max_length=self.max_length, device=self.device
            )
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                decision_index=batch["decision_index"],
                num_decisions=batch["num_decisions"],
                max_options=batch["max_options"],
                option_slot=batch["option_slot"],
            )
            for case, row in zip(chunk, logits):
                ids = case.option_ids
                probs.append({oid: row[i].item() for i, oid in enumerate(ids)})
        return probs
