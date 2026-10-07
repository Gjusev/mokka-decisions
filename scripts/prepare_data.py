"""Build the normalized dataset: download sources, split by group, expand.

Deterministic end to end: same inputs -> same instances, same hashes. Usage::

    python scripts/prepare_data.py --config configs/pilot.yaml --out data/processed/v1

Outputs: utterances/*.jsonl, instances/{train,dev,cal_temperature,cal_policy}.jsonl,
eval/*.jsonl, manifests/{sources,splits,revisions}.json. The test split is
materialized only as eval *specifications* (held-out pools), never written as
training-side instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mokka_decisions.contracts import DecisionCase, dump_jsonl  # noqa: E402
from mokka_decisions.data import (  # noqa: E402
    MASSIVE_LOCALES,
    SOURCE_META,
    label_is_oos,
    load_source,
)
from mokka_decisions.instances import (  # noqa: E402
    build_all_catalogs,
    expand_rows,
    make_case,
    make_cross_domain_probe,
    make_full_taxonomy_case,
)
from mokka_decisions.splitting import (  # noqa: E402
    assert_group_disjunction,
    assert_no_overlap,
    assign_splits,
    split_counts,
)
from mokka_decisions.taxonomies import build_catalog  # noqa: E402


def subsample(rows, budget: int, salt: str) -> list[dict]:
    """Deterministic subsample: order by hash(row id + salt), keep first N."""
    if budget >= len(rows):
        return rows
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())
    return ranked[:budget]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--out", default="data/processed/v1")
    parser.add_argument("--sources", nargs="*", default=["banking77", "massive", "clinc"])
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    out = Path(args.out)
    (out / "utterances").mkdir(parents=True, exist_ok=True)
    (out / "instances").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    budgets = data_cfg.get("budgets", {})
    clinc_oos_budget = data_cfg.get("clinc_oos_budget", 500)

    all_rows: list[dict] = []
    labels_by_source: dict[str, list[str]] = {}
    dataset_versions: dict[str, dict] = {}

    for source in args.sources:
        print(f"[{source}] downloading + normalising ...")
        rows = [r for r in load_source(source) if r["text"]]
        # leakage policy: exact-duplicate texts against the official test are
        # excluded from the non-test side (test is never modified)
        from mokka_decisions.splitting import _text_hash

        test_hashes = {
            _text_hash(r["text"]) for r in rows if r["official_split"] == "test"
        }
        before = len(rows)
        rows = [
            r
            for r in rows
            if r["official_split"] == "test" or _text_hash(r["text"]) not in test_hashes
        ]
        dropped = before - len(rows)
        labels = sorted({r["label"] for r in rows if not label_is_oos(r["label"])})
        labels_by_source[source] = labels
        dataset_versions[source] = {
            "n_rows": len(rows),
            "dropped_duplicate_vs_test": dropped,
            "n_labels": len(labels),
            "oos_rows": sum(1 for r in rows if label_is_oos(r["label"])),
            "official_splits": dict(Counter(r["official_split"] for r in rows)),
        }
        # split by group BEFORE any expansion; official test stays test
        rows = assign_splits(rows)
        assert_group_disjunction(rows)
        assert_no_overlap(rows, "train", "test")
        assert_no_overlap(rows, "dev", "test")
        print(f"[{source}] splits: {split_counts(rows)}")
        all_rows.extend(rows)
        dump_jsonl(rows, out / "utterances" / f"{source}.jsonl")

    # cross-source leakage guard (identical text in different sources is fine,
    # but the same group can never span splits)
    assert_group_disjunction(all_rows)

    catalogs = build_all_catalogs(labels_by_source)

    # per-split instance expansion with per-source budgets
    split_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        split_rows[row["split"]].append(row)

    for split in ("train", "dev", "cal_temperature", "cal_policy"):
        selected: list[dict] = []
        by_source = defaultdict(list)
        for row in split_rows[split]:
            by_source[row["source"]].append(row)
        for source, rows in by_source.items():
            budget_key = f"{source}_{split}" if split != "train" else source
            budget = budgets.get(budget_key, budgets.get(source, len(rows)))
            if source == "clinc" and split == "train":
                in_scope = [r for r in rows if not label_is_oos(r["label"])]
                oos = [r for r in rows if label_is_oos(r["label"])]
                selected.extend(subsample(in_scope, budget, f"{source}|{split}"))
                selected.extend(subsample(oos, clinc_oos_budget, f"{source}|oos|{split}"))
            else:
                selected.extend(subsample(rows, budget, f"{source}|{split}"))
        cases = expand_rows(selected, catalogs, split=split)
        dump_jsonl([c.to_dict() for c in cases], out / "instances" / f"{split}.jsonl")
        print(f"[{split}] {len(cases)} instances "
              f"({dict(Counter(c.language for c in cases))})")

    # ---- eval sets (deterministic specs; built from held-out pools) ----
    eval_specs = {}

    def build_eval(name, rows, builder):
        cases = [builder(r) for r in rows]
        dump_jsonl([c.to_dict() for c in cases], out / "eval" / f"{name}.jsonl")
        eval_specs[name] = {
            "n": len(cases),
            "from_split": "test" if rows and rows[0]["split"] == "test" else rows[0]["split"] if rows else "?",
            "languages": dict(Counter(c.language for c in cases)),
        }
        return cases

    test_rows = split_rows["test"]
    dev_rows = split_rows["dev"]

    # 1. candidate-set eval on the official test pools (banking + massive + clinc)
    for source in args.sources:
        rows = [r for r in test_rows if r["source"] == source and not label_is_oos(r["label"])]
        budget = data_cfg.get("eval_budgets", {}).get(source, len(rows))
        rows = subsample(rows, budget, f"eval|{source}")
        catalog = catalogs[source]
        build_eval(
            f"{source}_test_candidates",
            rows,
            lambda r, cat=catalog: make_case(r, cat, k=6, include_none=True, salt="test-v1"),
        )

    # 2. clinc OOS test: gold=none with clinc distractors
    oos_rows = [r for r in test_rows if r["source"] == "clinc" and label_is_oos(r["label"])]
    oos_rows = subsample(oos_rows, data_cfg.get("eval_budgets", {}).get("clinc_oos", 300), "eval|clinc_oos")
    build_eval(
        "clinc_oos_test",
        oos_rows,
        lambda r: make_case(r, catalogs["clinc"], k=6, include_none=True, salt="test-oos-v1"),
    )

    # 3. full-taxonomy eval (banking 77-way, from official test)
    bank_test = [r for r in test_rows if r["source"] == "banking77"]
    bank_test = subsample(bank_test, data_cfg.get("eval_budgets", {}).get("banking_full", 1000), "eval|bank_full")
    build_eval("banking77_test_full", bank_test, lambda r: make_full_taxonomy_case(r, catalogs["banking77"]))

    # 4. cross-domain OOS probes: utterance domain unseen for the option schema
    probe_rows = subsample(
        [r for r in test_rows if r["source"] == "massive" and r["language"] != "en"],
        data_cfg.get("eval_budgets", {}).get("xprobe", 300),
        "eval|xprobe",
    )
    build_eval("xprobe_massive_vs_clinc", probe_rows, lambda r: make_cross_domain_probe(r, catalogs["clinc"]))

    # 5. dev candidate-set (for model selection; fixed k=6 same as test)
    dev_sel = subsample(dev_rows, data_cfg.get("eval_budgets", {}).get("dev", 800), "eval|dev")
    dev_by_source = defaultdict(list)
    for r in dev_sel:
        dev_by_source[r["source"]].append(r)
    dev_cases = []
    for source, rows in dev_by_source.items():
        dev_cases.extend(
            make_case(r, catalogs[source], k=6, include_none=True, salt="dev-eval-v1") for r in rows
        )
    dump_jsonl([c.to_dict() for c in dev_cases], out / "eval" / "dev_eval.jsonl")
    eval_specs["dev_eval"] = {"n": len(dev_cases), "languages": dict(Counter(c.language for c in dev_cases))}

    # ---- manifests ----
    sources_manifest = {
        src: {
            **SOURCE_META[src],
            **dataset_versions[src],
            "labels": labels_by_source[src],
        }
        for src in args.sources
    }
    (out / "manifests" / "sources.json").write_text(
        json.dumps(sources_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    splits_manifest = {
        "counts": split_counts(all_rows),
        "per_source": {
            src: split_counts([r for r in all_rows if r["source"] == src]) for src in args.sources
        },
        "eval_specs": eval_specs,
        "guarantees": [
            "group-level hash split before expansion",
            "official test pools preserved as final test",
            "cross-locale MASSIVE groups share one split",
            "no identical text across train/test or dev/test",
        ],
    }
    (out / "manifests" / "splits.json").write_text(
        json.dumps(splits_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    data_hash = hashlib.sha256(
        json.dumps(
            {
                "sources": sources_manifest,
                "splits": {k: v for k, v in splits_manifest["counts"].items()},
                "eval_specs": eval_specs,
                "config": cfg,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (out / "manifests" / "revisions.json").write_text(
        json.dumps(
            {
                "data_hash": data_hash,
                "config_file": args.config,
                "description": "hash over sources+split counts+eval specs+config",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] data_hash={data_hash[:16]} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
