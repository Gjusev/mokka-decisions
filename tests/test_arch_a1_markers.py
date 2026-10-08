"""Marker bookkeeping in arch_a1 (count, expansion, option mapping).

Regression test for the n_markers bug found by the 2026-10-08 review:
ensure_markers stored a token id + 1 as the marker count, so later
expansions were silent no-ops and marker_ids() iterated ~vocab times.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mokka_decisions.arch_a1 import MAX_MARKERS, MarkerScorer


class FakeTokenizer:
    def __init__(self, base_vocab: int = 1000):
        self.base = base_vocab
        self.vocab = {f"tok{i}": i for i in range(base_vocab)}
        self.specials: list[str] = []

    def add_special_tokens(self, payload: dict) -> int:
        added = 0
        for tok in payload.get("additional_special_tokens", []):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
                self.specials.append(tok)
                added += 1
        return added

    def convert_tokens_to_ids(self, tok: str):
        return self.vocab.get(tok)

    def __len__(self) -> int:
        return len(self.vocab)


class FakeEncoder:
    def __init__(self):
        self.resized_to = None

    def resize_token_embeddings(self, n: int) -> None:
        self.resized_to = n


def make_model() -> MarkerScorer:
    m = MarkerScorer.__new__(MarkerScorer)  # skip heavyweight __init__
    m.tokenizer = FakeTokenizer()
    m.encoder = FakeEncoder()
    m.n_markers = 0
    return m


def test_ensure_markers_counts_and_resizes():
    m = make_model()
    m.ensure_markers(16)
    assert m.n_markers == 16, f"count, not token id: {m.n_markers}"
    assert len(m.tokenizer.specials) == 16
    assert m.tokenizer.specials[0] == "[O0]" and m.tokenizer.specials[15] == "[O15]"
    assert m.encoder.resized_to == len(m.tokenizer)


def test_expansion_after_first_batch():
    m = make_model()
    m.ensure_markers(16)
    m.ensure_markers(20)
    assert m.n_markers == 20
    assert len(m.tokenizer.specials) == 20, "expansion must add [O16..O19]"
    assert m.tokenizer.specials[19] == "[O19]"


def test_idempotent_and_shrink_noop():
    m = make_model()
    m.ensure_markers(16)
    vocab_before = len(m.tokenizer)
    m.ensure_markers(10)  # fewer than present: no-op
    m.ensure_markers(16)  # same: no-op
    assert m.n_markers == 16
    assert len(m.tokenizer) == vocab_before


def test_marker_ids_match_count_and_order():
    m = make_model()
    m.ensure_markers(16)
    ids = m.marker_ids()
    assert len(ids) == 16, f"marker_ids must return n_markers ids, got {len(ids)}"
    assert all(i is not None for i in ids)
    assert ids == sorted(ids), "ids must follow marker order [O0..O15]"


def test_cap_at_max_markers():
    m = make_model()
    m.ensure_markers(MAX_MARKERS + 10)
    assert m.n_markers == MAX_MARKERS
    assert len(m.marker_ids()) == MAX_MARKERS
