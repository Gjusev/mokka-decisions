"""Consolidate the typed-decisions comparison into one table.

Inputs (any present are included):
- runs/julia-research/reproduce-typed/results.json           (reproduced reference)
- runs/typed-decisions/mokka-base-protocol/results.json      (our zero-shot, frozen protocol)
- runs/kaggle-typed-arms-*/typed_arms_results.json           (adapted arms, kernel output)
- runs/kaggle-typed-a1*/typed_a1_results.json                (A1 architecture, if run)

Output: markdown table + JSON with explicit labels per row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    ("Julia-1 (reproduced, official script, CPU)", "reproduced_reference",
     ROOT / "runs/julia-research/reproduce-typed/results.json"),
    ("mokka-base zero-shot (frozen protocol)", "zero_shot_domain_shift",
     ROOT / "runs/typed-decisions/mokka-base-protocol/results.json"),
    ("mokka-base_fresh (typed-train)", "adapted_equal_data",
     ROOT / "runs/kaggle-typed-arms-final/typed_arms_results.json"),
    ("mokka A1 markers (typed-train)", "adapted_equal_data_a1",
     ROOT / "runs/kaggle-typed-a1/typed_a1_results.json"),
]


def main() -> int:
    rows = []
    for label, tag, path in CANDIDATES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if tag == "reproduced_reference":
            by_type = data["original_criteria"]["by_type"]
            total = sum(v["correct"] for v in by_type.values())
            rows.append({
                "label": label, "tag": tag, "overall": total / 2000,
                "overall_correct": f"{total}/2000",
                "choice": by_type["choice"]["accuracy"],
                "noul": by_type["noul"]["accuracy"],
                "score": by_type["score"]["accuracy"],
                "params_M": 144.3, "source": str(path),
            })
        elif tag == "zero_shot_domain_shift":
            by_type = data["by_type"]
            rows.append({
                "label": label, "tag": tag, "overall": data["overall_accuracy"],
                "overall_correct": data["overall_correct"],
                "choice": by_type["choice"]["accuracy"],
                "noul": by_type["noul"]["accuracy"],
                "score": by_type["score"]["accuracy"],
                "score_mae": data.get("score_mae_expected_value"),
                "params_M": 307.0, "source": str(path),
            })
        else:
            arms = data.get("test", data) if "test" in data else data
            if tag == "adapted_equal_data":
                for arm, info in data.items():
                    if arm.startswith("_") or not isinstance(info, dict) or "test" not in info:
                        continue
                    t = info["test"]
                    bt = t["by_type"]
                    rows.append({
                        "label": f"mokka-{arm} (typed-train)", "tag": tag,
                        "overall": t["overall_accuracy"], "overall_correct": t["overall"],
                        "choice": bt["choice"]["accuracy"], "noul": bt["noul"]["accuracy"],
                        "score": bt["score"]["accuracy"],
                        "score_mae": t.get("score_mae"),
                        "dev": info.get("best_dev_accuracy"),
                        "params_M": 307.0 if "base" in arm else 140.5,
                        "source": str(path),
                    })
            else:
                t = data["test"]
                bt = t["by_type"]
                rows.append({
                    "label": label, "tag": tag, "overall": t["overall_accuracy"],
                    "overall_correct": t["overall"],
                    "choice": bt["choice"]["accuracy"], "noul": bt["noul"]["accuracy"],
                    "score": bt["score"]["accuracy"],
                    "dev": data.get("best_dev_accuracy"), "params_M": 307.0,
                    "source": str(path),
                })

    out_dir = ROOT / "runs/typed-decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = ["| Runner | Label | Overall | Choice | Noul | Score | Params |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['tag']} | **{r['overall']:.4f}** ({r['overall_correct']}) "
            f"| {r['choice']:.3f} | {r['noul']:.3f} | {r['score']:.3f} | {r['params_M']:.0f}M |"
        )
    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
