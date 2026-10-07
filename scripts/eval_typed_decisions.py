"""Evaluate a Mokka checkpoint on the pinned typed-decisions benchmark.

Implements the frozen protocol (docs/typed-decisions-protocol.md): rows built
identically to Julia's official script, no none option, forced predictions,
accuracy per type + MAE (expected value) for score rows + NLL.

Usage:
    python scripts/eval_typed_decisions.py --release release/mokka-decisions-v0.1.0 \
        --out runs/typed-decisions/mokka-base [--device cuda]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
DATA_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
DATA_URL = (
    f"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/{REVISION}"
    "/all/test-00000-of-00001.parquet"
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_rows(parquet_path: Path):
    """Identical row construction to Julia's reproduce_typed.py."""
    import pyarrow.parquet as pq

    cases = pq.read_table(parquet_path).to_pylist()
    rows, metadata = [], []
    for case in cases:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): value for i, value in enumerate(criteria)}
            if not isinstance(criteria, dict):
                raise ValueError("Missing option descriptions")
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            rows.append(
                dict(state=state, question=question["instructions"], type=kind,
                     options=[criteria[key] for key in keys])
            )
            metadata.append(
                dict(id=case["id"] + ":" + question_id, type=kind, keys=keys,
                     gold=str(gold[question_id]["label"]))
            )
    counts = collections.Counter(r["type"] for r in rows)
    assert len(cases) == 400 and counts == {"choice": 600, "score": 800, "noul": 600}, counts
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="export dir (from export_release.py)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-check", action="store_true", help="verify batch==single on 50 rows")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / "test-00000-of-00001.parquet"
    if not parquet.exists():
        tmp = parquet.with_suffix(".partial")
        with urllib.request.urlopen(DATA_URL, timeout=180) as source, tmp.open("wb") as target:
            shutil.copyfileobj(source, target)
        tmp.replace(parquet)
    if digest(parquet) != DATA_SHA256:
        raise SystemExit("dataset sha256 mismatch — refusing to proceed")

    rows, metadata = build_rows(parquet)

    import torch

    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.export import load_exported_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scorer, _policy = load_exported_model(Path(args.release))
    scorer.device = device
    scorer.model = scorer.model.to(device)

    # token-length census (protocol: measured after freezing, no re-tuning)
    enc_lens = []
    for r in rows[:200]:
        ids = scorer.tokenizer(str(r["state"]), r["question"])["input_ids"]
        enc_lens.append(len(ids))
    enc_lens.sort()
    census = {"p50": enc_lens[len(enc_lens) // 2], "p99": enc_lens[int(0.99 * len(enc_lens))]}

    cases = [
        DecisionCase(
            id=m["id"], group_id=m["id"].split(":")[0], source="typed-decisions",
            language="en", domain="benchmark", state=str(r["state"]),
            question=r["question"], options=tuple(Option(id=f"o{i}", description=d) for i, d in enumerate(r["options"])),
            target_kind="single", target_option=f"o{m['keys'].index(m['gold'])}",
        )
        for r, m in zip(rows, metadata)
    ]

    # batch-vs-single check (protocol step 4)
    if args.batch_check:
        single = scorer.score(cases[:50], )
        batched = []
        for s in range(0, 50, 16):
            chunk = cases[s : s + 16]
            probs = scorer.score(chunk)
            for c, p in zip(chunk, probs):
                batched.append((c.id, max(p, key=p.get)))
        mism = sum(1 for (cid, cand), p in zip(batched, single) if max(p, key=p.get) != cand)
        print(f"batch-vs-single argmax mismatches on 50 rows: {mism}")

    t0 = time.time()
    all_probs = []
    for s in range(0, len(cases), 16):
        all_probs.extend(scorer.score(cases[s : s + 16]))
    elapsed = time.time() - t0

    stats = collections.defaultdict(lambda: dict(count=0, correct=0))
    mae_sum = 0.0
    mae_n = 0
    nll_sum = 0.0
    pred_path = out / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as fh:
        for case, meta, probs in zip(cases, metadata, all_probs):
            gold_idx = meta["keys"].index(meta["gold"])
            pred_idx = int(max(probs, key=probs.get)[1:])
            correct = pred_idx == gold_idx
            st = stats[meta["type"]]
            st["count"] += 1
            st["correct"] += int(correct)
            ordered = [probs[f"o{i}"] for i in range(len(meta["keys"]))]
            nll_sum += -__import__("math").log(max(ordered[gold_idx], 1e-12))
            if meta["type"] == "score":
                ev = sum(i * p for i, p in enumerate(ordered))
                mae_sum += abs(ev - gold_idx)
                mae_n += 1
            fh.write(json.dumps(dict(
                id=meta["id"], type=meta["type"], gold=meta["gold"],
                keys=meta["keys"], prediction=meta["keys"][pred_idx],
                correct=correct, probabilities=ordered,
            )) + "\n")
    for st in stats.values():
        st["accuracy"] = st["correct"] / st["count"]
    total_correct = sum(st["correct"] for st in stats.values())
    result = {
        "runner": "mokka (see release meta)",
        "protocol": "docs/typed-decisions-protocol.md v1.0",
        "dataset_sha256": DATA_SHA256,
        "dataset_revision": REVISION,
        "max_length": scorer.max_length,
        "state_token_census_first200": census,
        "device": device,
        "elapsed_s": round(elapsed, 1),
        "overall_accuracy": total_correct / len(cases),
        "overall_correct": f"{total_correct}/{len(cases)}",
        "by_type": dict(stats),
        "score_mae_expected_value": mae_sum / mae_n if mae_n else None,
        "mean_nll": nll_sum / len(cases),
        "release": str(args.release),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
