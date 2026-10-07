"""Batch JSONL decisions: one line per request, one line per answer.

    python examples/batch_decide.py --model-dir release/mokka-decisions-v0.1.0 \
        --input examples/batch_input.jsonl --output answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    from mokka_decisions.contracts import validate_request_payload
    from mokka_decisions.export import load_exported_model
    from mokka_decisions.policy import apply_policy

    scorer, policy = load_exported_model(Path(args.model_dir))
    requests = [
        validate_request_payload(json.loads(line))
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for start in range(0, len(requests), args.batch):
            chunk = requests[start : start + args.batch]
            probs = scorer.score(chunk)
            for case, p in zip(chunk, probs):
                response = apply_policy(
                    case,
                    p,
                    threshold=policy["threshold"],
                    model_revision=scorer.model_revision,
                    calibration_revision=policy.get("calibration_revision", ""),
                )
                fh.write(json.dumps(response.to_dict(), ensure_ascii=False) + "\n")
    print(f"{len(requests)} decisions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
