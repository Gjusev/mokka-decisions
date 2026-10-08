"""Preserve the original gold.probabilities of the typed-decisions TRAIN split.

E15 (2026-10-08 commission): the ordinal loss fabricates a soft target
exp(-|j-gold|/tau); the dataset itself carries teacher probabilities for
every question (5,977/6,000 multi-positive, sums ~1). This script emits,
per existing split (SAME case-id hash splits as prepare_typed_train.py —
imported, not reimplemented), one row per question:

    {"id": ..., "question_type": ..., "probabilities": [p_k in OPTION ORDER]}

Option order is the one prepare_typed_train uses to build options
(criteria insertion order; str(i) for list criteria; false/true for noul).
Verifies alignment with the existing cases/{split}.jsonl by id AND order,
support length, and normalization. train+dev only (dev rows are for
evaluation metrics, never for training). No test data is read here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_typed_train import split_of  # same deterministic splits


def question_keys(question: dict) -> tuple[list[str], dict]:
    criteria = question.get("criteria")
    kind = question["type"]
    if criteria is None and kind == "noul":
        criteria = {"false": "false", "true": "true"}
    if isinstance(criteria, list):
        criteria = {str(i): v for i, v in enumerate(criteria)}
    keys = ["false", "true"] if kind == "noul" else list(criteria)
    return keys, criteria


def main() -> int:
    out = Path("data/typed-v1/cases")
    parquet = json.loads((ROOT / "data/typed-v1/manifest.json").read_text(encoding="utf-8"))
    assert parquet["revision"], "manifest missing revision"
    import pyarrow.parquet as pq

    rows_by_split: dict[str, list[dict]] = {"train": [], "dev": []}
    stats = {"train": {"n": 0, "multi": 0, "bad_sum": 0, "bad_len": 0}, "dev": {"n": 0, "multi": 0, "bad_sum": 0, "bad_len": 0}}
    for case in pq.read_table(ROOT / "data/typed-v1/train-00000-of-00001.parquet").to_pylist():
        split = split_of(case["id"])
        if split not in rows_by_split:
            continue  # cal/test-adjacent splits are not emitted
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            keys, _ = question_keys(question)
            g = gold[question_id]
            if str(g["label"]) not in keys:
                continue  # same skip rule as prepare_typed_train
            probs = g.get("probabilities") or {}
            if set(probs) != set(keys):
                raise SystemExit(f"key mismatch {case['id']}:{question_id}: {sorted(probs)} vs {sorted(keys)}")
            vec = [float(probs[k]) for k in keys]
            st = stats[split]
            st["n"] += 1
            st["multi"] += int(sum(1 for v in vec if v > 0) > 1)
            st["bad_sum"] += int(abs(sum(vec) - 1.0) > 1e-4)
            st["bad_len"] += int(len(vec) != len(keys))
            rows_by_split[split].append({
                "id": f"{case['id']}:{question_id}",
                "question_type": question["type"],
                "probabilities": vec,
            })

    # alignment verification against the existing split files (ids AND order)
    for split, rows in rows_by_split.items():
        existing = [json.loads(l) for l in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        ids_new = [r["id"] for r in rows]
        ids_old = [r["id"] for r in existing]
        assert ids_new == ids_old, f"{split}: id alignment failed ({len(ids_new)} vs {len(ids_old)})"
        # option count per row must equal the probability vector length
        for r, old in zip(rows, existing):
            assert len(r["probabilities"]) == len(old["options"]), f"{r['id']}: support mismatch"
        with (out / f"probabilities_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows aligned+verified | stats {stats[split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
