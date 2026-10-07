"""A1: one sequence per decision, options marked inline, scores from markers.

Contrast with A0 (the released scorer): A0 encodes K pairs (state, question +
option) per decision — K encoder passes. A1 encodes ONE sequence::

    [CLS] state </s> question [O0] option_0 [O1] option_1 ... [O_{K-1}] ...

and scores each option from its marker token's hidden state with a shared
head. Cost per decision is one pass regardless of K (the hypothesis the suite
prompt asks to measure). Positional information: ModernBERT's relative
position encodings mean segment order can leak into scores; permutation
equivariance is therefore MEASURED (flip rate), not assumed — markers make
permutation -> probability permutation approximately, not exactly.

Markers: added special tokens "[O0]".."[O31]" (embedding resize, trainable).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import DecisionCase

MAX_MARKERS = 32


class MarkerScorer(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.n_markers = 0
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    # -- marker management -------------------------------------------------
    def ensure_markers(self, k: int) -> None:
        """Add [O_i] special tokens (idempotent); resize embeddings once."""
        if k <= self.n_markers:
            return
        new_tokens = [f"[O{i}]" for i in range(self.n_markers, min(k, MAX_MARKERS))]
        self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.n_markers = self.tokenizer.convert_tokens_to_ids(new_tokens[-1]) + 1

    def marker_ids(self) -> list[int]:
        return [self.tokenizer.convert_tokens_to_ids(f"[O{i}]") for i in range(self.n_markers)]

    # -- encoding ----------------------------------------------------------
    def max_length(self) -> int:
        cap = getattr(self.encoder.config, "max_position_embeddings", None)
        return min(1024, cap) if cap else 1024

    def build_sequence(self, case: DecisionCase) -> dict:
        """state + question + marked options -> tokenized single sequence."""
        self.ensure_markers(len(case.options))
        parts = [case.state, self.tokenizer.sep_token, case.question]
        for i, option in enumerate(case.options):
            parts += [f"[O{i}]", option.description]
        text = " ".join(p for p in parts if p)
        return self.tokenizer(
            text, truncation=True, max_length=self.max_length(), return_tensors="pt"
        )

    # -- forward -----------------------------------------------------------
    def forward(self, input_ids, attention_mask, marker_index, num_decisions, max_options):
        """marker_index: (total_markers, 2) rows of (decision, slot)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (S, L, H)
        marker_rows = hidden[marker_index[:, 0], marker_index[:, 1]]  # (M, H)
        scores = self.head(marker_rows).squeeze(-1)
        logits = torch.full(
            (num_decisions, max_options),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        )
        logits[marker_index[:, 0], marker_index[:, 1] - 0] = scores  # slot = col
        return logits

    def parameter_groups(self, encoder_lr: float, head_lr: float) -> list[dict]:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        groups = []
        if enc:
            groups.append({"params": enc, "lr": encoder_lr})
        if rest:
            groups.append({"params": rest, "lr": head_lr})
        return groups


def collate_a1(model: MarkerScorer, cases: Sequence[DecisionCase], device):
    """Batch marker sequences (pad to longest) + marker index + targets."""
    sents = []
    marker_index = []
    targets = []
    for d, case in enumerate(cases):
        enc = model.build_sequence(case)
        sents.append(enc)
        ids = enc["input_ids"][0].tolist()
        marker_token_ids = model.marker_ids()
        slots = []
        for i in range(len(case.options)):
            tok = marker_token_ids[i]
            pos = ids.index(tok) if tok in ids else None
            if pos is None:  # truncated away: unusable decision in this batch
                slots = None
                break
            slots.append(pos)
        if slots is None:
            targets.append(None)
            continue
        for slot, pos in enumerate(slots):
            marker_index.append((d, pos, slot))
        targets.append(case.option_ids.index(case.target_option) if case.target_option else None)
    # pad sequences
    maxlen = max(int(e["input_ids"].shape[1]) for e in sents)
    pad = model.tokenizer.pad_token_id or 0
    input_ids = torch.full((len(sents), maxlen), pad, dtype=torch.long)
    attention = torch.zeros((len(sents), maxlen), dtype=torch.long)
    for r, e in enumerate(sents):
        n = int(e["input_ids"].shape[1])
        input_ids[r, :n] = e["input_ids"][0]
        attention[r, :n] = e["attention_mask"][0]
    max_options = max(len(c.options) for c in cases)
    mi = torch.tensor([(d, p) for d, p, _ in marker_index], dtype=torch.long)
    slots = torch.tensor([s for _, _, s in marker_index], dtype=torch.long)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "marker_index": mi.to(device),
        "slots": slots.to(device),
        "num_decisions": len(cases),
        "max_options": max_options,
        # scatter helper for forward: build (decision, slot) -> row mapping
        "marker_rows": torch.tensor(marker_index, dtype=torch.long, device=device),
        "targets": torch.tensor(
            [t if t is not None else -100 for t in targets], dtype=torch.long, device=device
        ),
    }


def forward_a1(model: MarkerScorer, batch: dict) -> torch.Tensor:
    outputs = model.encoder(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    hidden = outputs.last_hidden_state
    rows = batch["marker_rows"]
    marker_h = hidden[rows[:, 0], rows[:, 1]]
    scores = model.head(marker_h).squeeze(-1)
    logits = torch.full(
        (batch["num_decisions"], batch["max_options"]),
        torch.finfo(scores.dtype).min,
        dtype=scores.dtype,
        device=scores.device,
    )
    logits[rows[:, 0], rows[:, 2]] = scores
    return logits
