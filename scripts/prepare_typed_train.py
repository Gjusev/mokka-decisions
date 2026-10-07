"""Build decision cases from the typed-decisions TRAIN split.

Purpose: the *adapted* arm of the Julia comparison (equal-adaptation budget)
and, later, same-data comparisons for A0/A1/A2. The pinned TEST parquet is
never touched here.

Rows are constructed with the same rules as the frozen eval adapter
(json.dumps state, criteria order preserved, no `none` added, boolean
descriptions kept as written, ordinal score keys str(i)).

Splits: hash of the ORIGINAL CASE id (before per-question expansion), so
questions from one case never straddle splits. Output: data/typed-v1/.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
TRAIN_URL = (
    f"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/{REVISION}"
    "/all/train-00000-of-00001.parquet"
)


def split_of(group_id: str, ratios=(0.8, 0.1, 0.1)) -> str:
    u = int.from_bytes(hashlib.sha256(f"typed-split-v1|{group_id}".encode()).digest()[:8], "big") / 2**64
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "dev"
    return "cal"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/typed-v1")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / "train-00000-of-00001.parquet"
    if not parquet.exists():
        tmp = parquet.with_suffix(".partial")
        with urllib.request.urlopen(TRAIN_URL, timeout=180) as source, tmp.open("wb") as target:
            target.write(source.read())
        tmp.replace(parquet)

    import pyarrow.parquet as pq

    from mokka_decisions.contracts import DecisionCase, Option, dump_jsonl

    cases = pq.read_table(parquet).to_pylist()
    rows_by_split: dict[str, list[DecisionCase]] = collections.defaultdict(list)
    qtypes = collections.Counter()
    for case in cases:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        state_text = json.dumps(state, ensure_ascii=False)
        split = split_of(case["id"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            qtypes[kind] += 1
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): value for i, value in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            if str(gold[question_id]["label"]) not in keys:
                continue  # gold outside keys (shouldn't happen; count below)
            options = tuple(Option(id=f"o{i}", description=criteria[k]) for i, k in enumerate(keys))
            gold_idx = keys.index(str(gold[question_id]["label"]))
            rows_by_split[split].append(
                DecisionCase(
                    id=f"{case['id']}:{question_id}",
                    group_id=case["id"],
                    source="typed-decisions-train",
                    language="en",
                    domain="benchmark",
                    state=state_text,
                    question=question["instructions"],
                    options=options,
                    target_kind="single",
                    target_option=f"o{gold_idx}",
                    label_origin="benchmark-gold",
                    split=split if split != "cal" else "cal_temperature",
                )
            )

    counts = {}
    (out / "cases").mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        dump_jsonl([r.to_dict() for r in rows], out / "cases" / f"{split}.jsonl")
        counts[split] = len(rows)
        print(f"{split}: {len(rows)} question rows")
    manifest = {
        "source": "LocalLLaMA/typed-decisions (train split only)",
        "revision": REVISION,
        "train_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "counts": counts,
        "question_types": dict(qtypes),
        "splits": "hash of original case id before question expansion",
        "construction": "identical to frozen eval adapter; no none added",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
