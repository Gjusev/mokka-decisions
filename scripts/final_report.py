"""Consolidated final report: identical IDs/options across all backends.

Loads saved prediction rows (kernel outputs + local runs), verifies every
backend saw exactly the same case IDs per set, computes metrics v2 (incl.
multiclass Brier), applies each backend's own fitted threshold where
available, and emits a markdown table + machine-readable JSON.

Usage:
    python scripts/final_report.py --data data/processed/v1 \
        --mokka-rows runs/kaggle-final/rows \
        --extra-rows runs/kaggle-final-baselines/rows runs/local-baselines-full/rows \
        --policies runs/kaggle-final/policy.json runs/kaggle-final-baselines/baseline_policies.json \
        --out runs/final-report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mokka_decisions.backends.base import BackendRow  # noqa: E402
from mokka_decisions.contracts import DecisionCase  # noqa: E402
from mokka_decisions.evaluate import (  # noqa: E402
    bootstrap_accuracy_ci,
    oos_false_acceptance,
    summarize,
    summarize_by,
)

# xprobe label ambiguity: gold=none but the MASSIVE intent exists verbatim in
# the CLINC taxonomy (exact-name rule; applied identically to every backend)
AMBIGUOUS_RULE = "massive intent name in clinc taxonomy"


def load_rows(path: Path) -> list[BackendRow]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows.append(BackendRow(**{k: v for k, v in d.items() if k in BackendRow.__dataclass_fields__}))
    return rows


def backend_of(path: Path) -> str:
    return path.stem  # convention: <set>_<backend>.jsonl


def set_of(path: Path) -> str:
    stem = path.stem
    for known in (
        "banking77_test_candidates", "banking77_test_full", "massive_test_candidates",
        "clinc_test_candidates", "clinc_oos_test", "xprobe_massive_vs_clinc", "dev_eval",
    ):
        if stem.startswith(known + "_"):
            return known
    raise SystemExit(f"cannot parse set name from {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/v1")
    parser.add_argument("--mokka-rows", default="runs/kaggle-final/rows")
    parser.add_argument("--extra-rows", nargs="*", default=[])
    parser.add_argument("--policies", nargs="*", default=[])
    parser.add_argument("--out", default="runs/final-report")
    args = parser.parse_args()

    policies: dict[str, dict] = {}
    for pol_path in args.policies:
        p = Path(pol_path)
        if not p.exists():
            continue
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if "threshold" in loaded:  # single policy file (mokka)
            policies["mokka_option_scorer"] = loaded
        else:  # baseline_policies.json keyed by backend
            policies.update(loaded)

    eval_dir = Path(args.data) / "eval"
    row_dirs = [Path(args.mokka_rows)] + [Path(d) for d in args.extra_rows]
    files = [f for d in row_dirs if d.exists() for f in sorted(d.glob("*.jsonl"))]

    by_set: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in files:
        by_set[set_of(f)][backend_of(f)] = f

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    table: dict[str, dict[str, dict]] = {}

    for set_name in sorted(by_set):
        eval_file = eval_dir / f"{set_name}.jsonl"
        if not eval_file.exists():
            continue
        cases = [DecisionCase.from_dict(json.loads(ln)) for ln in eval_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        case_ids = [c.id for c in cases]
        set_report = {}
        for backend_name, rows_file in sorted(by_set[set_name].items()):
            rows = load_rows(rows_file)
            ids = [r.case_id for r in rows]
            if ids != case_ids:
                set_report[backend_name] = {
                    "id_mismatch": True, "rows": len(rows), "cases": len(cases),
                    "note": "not directly comparable (different sample)",
                }
                continue
            threshold = None
            if backend_name in policies:
                threshold = policies[backend_name].get("threshold")
            s = summarize(rows, cases, threshold=threshold)
            s["by_language"] = summarize_by(rows, cases, "language", threshold=threshold)
            s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=300)
            if any(c.target_option == "none" for c in cases):
                s["oos"] = oos_false_acceptance(rows, cases, threshold=threshold or 0.0)
            set_report[backend_name] = s

        # xprobe ambiguity view: same metrics excluding exact-name collisions
        if set_name == "xprobe_massive_vs_clinc":
            massive_labels = {}
            utt = Path(args.data) / "utterances" / "massive.jsonl"
            if utt.exists():
                for ln in utt.read_text(encoding="utf-8").splitlines():
                    r = json.loads(ln)
                    massive_labels[r["id"]] = r["label"]
            sources_file = Path(args.data) / "manifests" / "sources.json"
            clinc_labels = set(json.loads(sources_file.read_text(encoding="utf-8"))["clinc"]["labels"])
            keep = [
                i for i, c in enumerate(cases)
                if massive_labels.get(c.id.replace("-xprobe", "")) not in clinc_labels
            ]
            for backend_name, rows_file in sorted(by_set[set_name].items()):
                rows = load_rows(rows_file)
                if [r.case_id for r in rows] != case_ids:
                    continue
                sub_rows = [rows[i] for i in keep]
                sub_cases = [cases[i] for i in keep]
                thr = policies.get(backend_name, {}).get("threshold")
                set_report[backend_name]["ambiguity_clean"] = summarize(sub_rows, sub_cases, threshold=thr)
                set_report[backend_name]["ambiguity_clean_n"] = len(keep)
            set_report["_ambiguity_rule"] = AMBIGUOUS_RULE

        report[set_name] = set_report
        table[set_name] = {
            b: {
                "acc": s.get("forced_accuracy"),
                "f1": s.get("macro_f1"),
                "nll": s.get("nll"),
                "brier": s.get("brier"),
                "ece": s.get("ece"),
                "coverage": s.get("coverage"),
                "risk": s.get("selective_risk"),
                "oos_fa": (s.get("oos") or {}).get("false_acceptance_rate") if s.get("oos") else None,
                "p50_ms": s.get("latency_ms_p50"),
                "mismatch": s.get("id_mismatch", False),
            }
            for b, s in set_report.items() if isinstance(s, dict) and "forced_accuracy" in s
        }

    (out / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # markdown table
    lines = ["| Set | Backend | Acc | F1 | NLL | Brier | ECE | Cov | Risk | OOS-FA | p50ms |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for set_name, backends in table.items():
        for b, s in backends.items():
            def fmt(x, nd=3):
                return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
            lines.append(
                f"| {set_name} | {b}{'' if not s['mismatch'] else ' (⚠ sample difiere)'} | "
                f"{fmt(s['acc'])} | {fmt(s['f1'])} | {fmt(s['nll'])} | {fmt(s['brier'])} | "
                f"{fmt(s['ece'])} | {fmt(s['coverage'])} | {fmt(s['risk'], 4)} | {fmt(s['oos_fa'])} | {fmt(s['p50_ms'], 0)} |"
            )
    (out / "final_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
