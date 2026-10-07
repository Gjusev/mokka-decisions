"""Group-level split assignment, before any expansion or augmentation.

Split assignment is a pure function of ``group_id`` (sha256 -> uniform [0,1)).
Two properties fall out for free:

* determinism: re-running the build produces identical splits;
* cross-locale consistency: MASSIVE translations share a ``group_id``, so the
  German and Spanish versions of one utterance can never land on opposite
  sides of a split.

Official test pools are never re-split: they map straight to ``test`` and are
opened once at the end. Everything below assumes the caller already verified
that no group appears in both the official test pool and the training pools.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence

from .contracts import SPLITS

# Ratios over the non-test pool: train / dev / cal_temperature / cal_policy
DEFAULT_RATIOS = {"train": 0.80, "dev": 0.08, "cal_temperature": 0.06, "cal_policy": 0.06}

RATIO_ORDER = ("train", "dev", "cal_temperature", "cal_policy")


def split_for_group(group_id: str, ratios: Mapping[str, float] = DEFAULT_RATIOS) -> str:
    """Deterministic split assignment from the group id alone."""
    for name in RATIO_ORDER:
        if name not in ratios:
            raise ValueError(f"missing ratio for {name!r}")
    total = sum(ratios[name] for name in RATIO_ORDER)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"ratios sum to {total!r}, expected ~1.0")
    digest = hashlib.sha256(f"mokka-split-v1|{group_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    acc = 0.0
    for name in RATIO_ORDER:
        acc += ratios[name]
        if u < acc:
            return name
    return RATIO_ORDER[-1]


def assign_splits(
    rows: Iterable[Mapping],
    *,
    test_marker: str = "test",
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
) -> list[dict]:
    """Return rows with an assigned ``split``; official test passes through."""
    out: list[dict] = []
    for row in rows:
        new = dict(row)
        official = new.get("official_split") or new.get("split")
        if official == test_marker:
            new["split"] = "test"
        else:
            new["split"] = split_for_group(new["group_id"], ratios)
        out.append(new)
    return out


def assert_group_disjunction(rows: Sequence[Mapping]) -> None:
    """Fail loudly if any group_id spans more than one split (leakage guard)."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        seen.setdefault(row["group_id"], set()).add(row["split"])
    leaks = {g: s for g, s in seen.items() if len(s) > 1}
    if leaks:
        sample = dict(list(leaks.items())[:5])
        raise AssertionError(f"group_id appears in multiple splits: {sample}")


def assert_no_overlap(rows: Sequence[Mapping], split_a: str, split_b: str) -> None:
    """Also guard exact and near-duplicate text across two splits (by text hash)."""
    texts_a = {
        _text_hash(r["text"]) for r in rows if r["split"] == split_a and r.get("text")
    }
    for r in rows:
        if r["split"] == split_b and r.get("text"):
            if _text_hash(r["text"]) in texts_a:
                raise AssertionError(
                    f"identical text in both {split_a} and {split_b}: group {r['group_id']}"
                )


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def split_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    counts = {name: 0 for name in SPLITS}
    for row in rows:
        counts[row["split"]] += 1
    return counts
