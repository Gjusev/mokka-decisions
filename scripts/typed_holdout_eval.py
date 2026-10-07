"""FROZEN holdout evaluation: typed-decisions pinned test, per-domain partitions.

Frozen BEFORE any finalist selection (2026-10-08), per the focused-research
commission. Exposure declaration: the aggregate of all/test has been
consulted during arm development (Julia reproduction, zero-shot, A0 arms);
PER-DOMAIN results have never been computed or used for any decision. This
script fixes the evaluation so the finalist cannot be tuned per-domain.

Domains (pinned revision c76749ec…): agent_trace_observability,
customer_service, invoice_processing, security_incidents — 100 cases each
(= the 400 of all/test; partition, not new data).

Protocol (identical construction to the frozen v1.0 adapter):
json.dumps(state), criteria order preserved, no none, argmax accuracy by
type, MAE by expected value for score, temperature 1.0, max_length 512
(A0) / model max (A1), batch 16.

Usage:
    python scripts/typed_holdout_eval.py --release <export-dir-or-arm-ckpt> \
        --encoder jhu-clsp/mmBERT-base --arch a0 --out runs/holdout/<name>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
DOMAINS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)


def domain_rows(cfg: str) -> list[tuple]:
    """(state_json, question, kind, options, gold, case_id, qid) per domain test."""
    import pyarrow.parquet as pq

    cache = ROOT / ".cache" / "typed-holdout-probe"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{cfg}-test.parquet"
    if not path.exists():
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/{REVISION}/{cfg}/test-00000-of-00001.parquet",
            path,
        )
    rows = []
    for case in pq.read_table(path).to_pylist():
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            rows.append((json.dumps(state, ensure_ascii=False), question["instructions"],
                         kind, [criteria[k] for k in keys], str(gold[question_id]["label"]),
                         case["id"], question_id, keys))
    return rows


def evaluate_a0(release_dir: Path, device: str) -> dict:
    import torch

    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.export import load_exported_model

    scorer, _ = load_exported_model(release_dir)
    scorer.device = device
    scorer.model = scorer.model.to(device)
    scorer.max_length = 512
    scorer.temperature = 1.0

    results = {}
    for cfg in DOMAINS:
        rows = domain_rows(cfg)
        cases = [DecisionCase(
            id=f"{cid}:{qid}", group_id=cid, source="typed-holdout", language="en",
            domain=cfg, state=sj, question=q,
            options=tuple(Option(id=f"o{i}", description=d) for i, d in enumerate(opts)),
            target_kind="single", target_option=f"o{keys.index(gold)}",
        ) for sj, q, kind, opts, gold, cid, qid, keys in rows]
        probs_all = []
        for s in range(0, len(cases), 16):
            probs_all.extend(scorer.score(cases[s : s + 16]))
        stats = collections.defaultdict(lambda: dict(count=0, correct=0))
        mae = n_mae = 0
        for case, (sj, q, kind, opts, gold, cid, qid, keys), probs in zip(cases, rows, probs_all):
            gold_idx = keys.index(gold)
            pred_idx = int(max(probs, key=probs.get)[1:])
            st = stats[kind]
            st["count"] += 1
            st["correct"] += int(pred_idx == gold_idx)
            if kind == "score":
                ordered = [probs[f"o{i}"] for i in range(len(keys))]
                mae += abs(sum(i * p for i, p in enumerate(ordered)) - gold_idx)
                n_mae += 1
        for st in stats.values():
            st["accuracy"] = st["correct"] / st["count"]
        total = sum(st["correct"] for st in stats.values())
        n = sum(st["count"] for st in stats.values())
        results[cfg] = {
            "overall_accuracy": total / n, "overall_correct": f"{total}/{n}",
            "by_type": dict(stats), "score_mae": mae / n_mae if n_mae else None,
        }
    return results


def evaluate_a1(ckpt: Path, encoder: str, device: str) -> dict:
    import torch

    from mokka_decisions.arch_a1 import MarkerScorer, collate_a1, forward_a1
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import load_checkpoint

    model = MarkerScorer(encoder)
    load_checkpoint(model, ckpt)
    model = model.to(device).eval()

    results = {}
    for cfg in DOMAINS:
        rows = domain_rows(cfg)
        cases = [DecisionCase(
            id=f"{cid}:{qid}", group_id=cid, source="typed-holdout", language="en",
            domain=cfg, state=sj, question=q,
            options=tuple(Option(id=f"o{i}", description=d) for i, d in enumerate(opts)),
            target_kind="single", target_option=f"o{keys.index(gold)}",
        ) for sj, q, kind, opts, gold, cid, qid, keys in rows]
        stats = collections.defaultdict(lambda: dict(count=0, correct=0))
        skipped = 0
        for s in range(0, len(cases), 8):
            chunk = cases[s : s + 8]
            batch = collate_a1(model, chunk, device)
            logits = forward_a1(model, batch).float()
            preds = logits.argmax(-1).tolist()
            for i, ((sj, q, kind, opts, gold, cid, qid, keys), case) in enumerate(zip(rows[s : s + 8], chunk)):
                st = stats[kind]
                if batch["targets"][i].item() == -100:
                    skipped += 1
                    st["count"] += 1
                    continue
                gold_idx = keys.index(gold)
                st["count"] += 1
                st["correct"] += int(preds[i] == gold_idx)
        for st in stats.values():
            st["accuracy"] = st["correct"] / st["count"] if st["count"] else None
        total = sum(st["correct"] for st in stats.values())
        n = sum(st["count"] for st in stats.values())
        results[cfg] = {
            "overall_accuracy": total / n, "overall_correct": f"{total}/{n}",
            "by_type": dict(stats), "skipped_truncation": skipped,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="A0 export dir OR A1 checkpoint .safetensors")
    parser.add_argument("--arch", choices=["a0", "a1"], default="a0")
    parser.add_argument("--encoder", default="jhu-clsp/mmBERT-base")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    release = Path(args.release)
    results = (
        evaluate_a0(release, device) if args.arch == "a0"
        else evaluate_a1(release, args.encoder, device)
    )
    out = {
        "frozen_protocol": "typed-holdout v1.0 (per-domain partitions of pinned all/test; aggregate exposure declared)",
        "revision": REVISION,
        "domains": list(DOMAINS),
        "arch": args.arch,
        "checkpoint": str(release),
        "checkpoint_sha256": hashlib.sha256(release.read_bytes()).hexdigest()[:16] if release.is_file() else "",
        "results": results,
    }
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "holdout_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
