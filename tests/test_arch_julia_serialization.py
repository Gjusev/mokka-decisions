"""Fidelity check: our julia_sequence port vs the upstream implementation.

Compares ids/markers/qtype token-for-token on typed-style rows using the SAME
tokenizer. Skips (not fails) when the upstream source tree is absent.

Also checks model construction parity: JuliaArchScorer on mmBERT-small must
have the same head/type_emb/scorer structure and dimensions as
JuliaDecisionModel (head_layers=2, width 768, scorer LN->L->GELU->L(1)).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
UPSTREAM = ROOT / "runs/julia-research/upstream-source"

ENCODER = "jhu-clsp/mmBERT-small"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ENCODER)


def sample_rows():
    return [
        dict(state=json.dumps({"k": "v", "n": 3}, ensure_ascii=False),
             question="Which carrier meets the deadline?",
             options=["AlphaFreight", "BetaExpress", "CargoJet"], type="choice"),
        dict(state=json.dumps({"readings": {"t": 84.1}}),
             question="Is maintenance overdue?", options=["false", "true"], type="noul"),
        dict(state="plain text state",
             question="Rate the urgency.",
             options=["low: none", "mid: one", "high: two", "urgent: three"], type="score"),
    ]


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream Julia source not present")
def test_sequence_matches_upstream(tokenizer):
    from mokka_decisions.arch_julia import julia_sequence

    sys.path.insert(0, str(UPSTREAM))
    from julia.data import sequence as upstream_sequence

    for row in sample_rows():
        ours = julia_sequence(tokenizer, row["state"], row["question"], row["options"], row["type"],
                              max_length=1024, head_length=256)
        theirs = upstream_sequence(tokenizer, row, 1024, 256)
        assert ours["ids"] == theirs["ids"], f"ids diverge for {row['question'][:30]}"
        assert ours["markers"] == theirs["markers"]
        assert ours["qtype"] == theirs["qtype"]


@pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream Julia source not present")
def test_model_structure_matches_upstream():
    import torch
    from mokka_decisions.arch_julia import JuliaArchScorer

    sys.path.insert(0, str(UPSTREAM))
    from julia.model import JuliaDecisionModel

    ours = JuliaArchScorer(ENCODER)
    theirs = JuliaDecisionModel.from_backbone(ENCODER)
    assert ours.encoder.config.hidden_size == theirs.encoder.config.hidden_size
    assert len(ours.head.layers) == len(theirs.head.layers) == 2
    l_ours, l_theirs = ours.head.layers[0], theirs.head.layers[0]
    assert type(l_ours) is type(l_theirs)
    assert l_ours.self_attn.num_heads == l_theirs.self_attn.num_heads
    assert l_ours.linear1.out_features == l_theirs.linear1.out_features
    assert ours.type_emb.num_embeddings == theirs.type_emb.num_embeddings == 3
    assert [type(m) for m in ours.scorer] == [type(m) for m in theirs.scorer]
    assert ours.scorer[1].out_features == theirs.scorer[1].out_features
    # forward parity on random input (both random init -> just shape/dtype contract)
    B, L, K = 2, 64, 4
    batch = dict(
        input_ids=torch.randint(0, 1000, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        marker_pos=torch.tensor([[5, 11, 17, 23]] * B),
        marker_mask=torch.ones(B, K, dtype=torch.bool),
        qtype=torch.tensor([1, 2]),
    )
    with torch.no_grad():
        out = ours(**batch)
    assert out.shape == (B, K) and out.dtype is torch.float32
