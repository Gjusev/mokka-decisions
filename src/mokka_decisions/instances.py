"""Candidate-set expansion: utterances -> decision instances.

Everything here is seeded and deterministic: the distractor sample, the option
count K and the option order are pure functions of (group_id, split, salt), so
rebuilding the data reproduces identical instances, and a differently-salted
permutation of the same case is available for the order-invariance probe.

Splitting happens *before* this module runs (see splitting.py) — expansion
never moves a case across splits.
"""

from __future__ import annotations

import hashlib
import random
from typing import Mapping, Sequence

from .contracts import NONE_ID, DecisionCase, Option
from .splitting import _text_hash  # noqa: PLC2701 - shared helper
from .taxonomies import DOMAIN_QUESTION, build_catalog


def _rng(group_id: str, salt: str) -> random.Random:
    seed = int.from_bytes(
        hashlib.sha256(f"mokka-inst-v1|{group_id}|{salt}".encode("utf-8")).digest()[:8],
        "big",
    )
    return random.Random(seed)


def _sample_distractors(
    rng: random.Random,
    catalog: Mapping[str, Option],
    gold: str,
    n: int,
) -> list[str]:
    pool = [oid for oid in catalog if oid not in (gold, NONE_ID)]
    if n > len(pool):
        n = len(pool)
    return rng.sample(pool, n)


def make_case(
    row: Mapping,
    catalog: Mapping[str, Option],
    *,
    k: int,
    include_none: bool,
    salt: str = "train",
    question: str | None = None,
) -> DecisionCase:
    """Build one decision case for an utterance row.

    Out-of-scope rows (label 'oos') always get gold=NONE_ID and none included.
    """
    rng = _rng(row["group_id"], salt)
    gold = NONE_ID if _is_oos(row) else row["label"]
    if gold not in catalog:
        raise KeyError(f"{row['id']}: label {gold!r} missing from catalog")

    if gold == NONE_ID:
        n_distractors = k - 1  # none itself occupies one slot
    else:
        n_distractors = k - 1 - (1 if include_none else 0)
    ids = _sample_distractors(rng, catalog, gold, n_distractors)
    if include_none or gold == NONE_ID:
        ids.append(NONE_ID)
    if gold != NONE_ID:
        ids.append(gold)
    rng.shuffle(ids)  # seeded permutation

    options = tuple(catalog[oid] for oid in ids)
    return DecisionCase(
        id=f"{row['id']}-k{k}",
        group_id=row["group_id"],
        source=row["source"],
        language=row["language"],
        domain=row["domain"],
        state=row["text"],
        question=question or DOMAIN_QUESTION[row["source"]],
        options=options,
        target_kind="single",
        target_option=gold,
        label_origin=row.get("label_origin", "public-dataset"),
        split=row.get("split"),
    )


def make_full_taxonomy_case(row: Mapping, catalog: Mapping[str, Option]) -> DecisionCase:
    """Options = the entire taxonomy (+none). Used for fixed-taxonomy eval."""
    ids = [oid for oid in catalog if oid != NONE_ID] + [NONE_ID]
    gold = NONE_ID if _is_oos(row) else row["label"]
    return DecisionCase(
        id=f"{row['id']}-full",
        group_id=row["group_id"],
        source=row["source"],
        language=row["language"],
        domain=row["domain"],
        state=row["text"],
        question=DOMAIN_QUESTION[row["source"]],
        options=tuple(catalog[oid] for oid in ids),
        target_kind="single",
        target_option=gold,
        label_origin=row.get("label_origin", "public-dataset"),
        split=row.get("split"),
    )


def make_cross_domain_probe(
    row: Mapping,
    foreign_catalog: Mapping[str, Option],
    *,
    k: int = 6,
    salt: str = "probe",
) -> DecisionCase:
    """An utterance scored against a taxonomy it never belongs to; gold=none.

    Measures 'decisions over domains never seen with this schema': the option
    descriptions are familiar, the utterance domain is not.
    """
    rng = _rng(row["group_id"], salt)
    ids = _sample_distractors(rng, foreign_catalog, NONE_ID, k - 1) + [NONE_ID]
    rng.shuffle(ids)
    return DecisionCase(
        id=f"{row['id']}-xprobe",
        group_id=row["group_id"],
        source=f"{row['source']}->probe",
        language=row["language"],
        domain=row["domain"],
        state=row["text"],
        question=DOMAIN_QUESTION["clinc"],
        options=tuple(foreign_catalog[oid] for oid in ids),
        target_kind="none",
        target_option=NONE_ID,
        label_origin="constructed-probe",
        split=row.get("split"),
    )


def permute_case(case: DecisionCase, seed: int) -> DecisionCase:
    """Same case, different option order (order-invariance probe)."""
    rng = random.Random(seed)
    order = list(range(len(case.options)))
    rng.shuffle(order)
    options = tuple(case.options[i] for i in order)
    return DecisionCase(
        id=f"{case.id}-perm{seed}",
        group_id=case.group_id,
        source=case.source,
        language=case.language,
        domain=case.domain,
        state=case.state,
        question=case.question,
        options=options,
        target_kind=case.target_kind,
        target_option=case.target_option,
        label_origin=case.label_origin,
        split=case.split,
    )


def _is_oos(row: Mapping) -> bool:
    label = str(row.get("label", "")).strip().lower()
    return label in {"oos", "out_of_scope"}


def expand_rows(
    rows: Sequence[Mapping],
    catalogs: Mapping[str, Mapping[str, Option]],
    *,
    split: str,
    train_k_range: tuple[int, int] = (4, 8),
    fixed_k: int = 6,
    include_none_prob: float = 0.5,
) -> list[DecisionCase]:
    """Expand utterance rows of one split into decision instances.

    train: K varies over train_k_range, none appears with include_none_prob
    (always for OOS rows). Other splits: fixed K, none always present.
    """
    cases: list[DecisionCase] = []
    is_train = split == "train"
    for row in rows:
        catalog = catalogs[row["source"]]
        if _is_oos(row) and row["source"] != "clinc":
            continue  # only clinc carries OOS labels
        if is_train:
            rng = _rng(row["group_id"], f"k|{split}")
            k = rng.randint(*train_k_range)
            include_none = rng.random() < include_none_prob or _is_oos(row)
        else:
            k = fixed_k
            include_none = True
        cases.append(
            make_case(
                row,
                catalog,
                k=k,
                include_none=include_none,
                salt=f"{split}-v1",
            )
        )
    return cases


def build_all_catalogs(labels_by_source: Mapping[str, list[str]]) -> dict[str, dict[str, Option]]:
    return {src: build_catalog(src, labels) for src, labels in labels_by_source.items()}
