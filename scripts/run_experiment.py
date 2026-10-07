"""Local experiment runner: baselines that need no GPU (tfidf_lr).

Usage: python scripts/run_experiment.py --data data/processed/v1 --out runs/local-baselines
Writes rows + summaries compatible with the Kaggle-side harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mokka_decisions.backends.linear import LinearBackend  # noqa: E402
from mokka_decisions.contracts import DecisionCase, load_jsonl  # noqa: E402
from mokka_decisions.evaluate import (  # noqa: E402
    bootstrap_accuracy_ci,
    dump_rows,
    run_backend,
    summarize,
    summarize_by,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/v1")
    parser.add_argument("--out", default="runs/local-baselines")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    for utt in sorted((data / "utterances").glob("*.jsonl")):
        train_rows.extend(r for r in load_jsonl(str(utt)) if r.get("split") == "train")
    backend = LinearBackend(train_rows)
    print(f"tfidf_lr fitted on {len(train_rows)} utterations, {len(backend.classes_)} classes")

    for eval_file in sorted((data / "eval").glob("*.jsonl")):
        name = eval_file.stem
        cases = [DecisionCase.from_dict(r) for r in load_jsonl(str(eval_file))]
        rows = run_backend(backend, cases)
        dump_rows(rows, out / "rows" / f"{name}_{backend.name}.jsonl")
        s = summarize(rows, cases)
        s["by_language"] = summarize_by(rows, cases, "language")
        s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=300)
        (out / "results" / f"{name}_{backend.name}.json").parent.mkdir(exist_ok=True)
        (out / "results" / f"{name}_{backend.name}.json").write_text(
            json.dumps(s, indent=2), encoding="utf-8"
        )
        print(f"{name}: forced_acc={s['forced_accuracy']:.4f} macro_f1={s['macro_f1']:.4f} "
              f"p50={s['latency_ms_p50']:.1f}ms errors={s['backend_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
