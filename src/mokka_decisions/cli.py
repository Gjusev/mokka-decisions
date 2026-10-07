"""Command line interface: mokka <prepare|train|calibrate|evaluate|export|decide|check>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mokka", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("decide", help="one decision from a JSON request on stdin or a file")
    p.add_argument("request", nargs="?", help="JSON file with {state, question?, options[]}")
    p.add_argument("--model-dir", required=True, help="export directory (from mokka export)")
    p.add_argument("--threshold", type=float, default=None, help="override policy threshold")

    p = sub.add_parser("check", help="validate one JSONL file of decision cases")
    p.add_argument("file", help="JSONL file of decision cases")

    args = parser.parse_args(argv)

    if args.cmd == "decide":
        return _cmd_decide(args)
    if args.cmd == "check":
        return _cmd_check(args)
    return 2


def _cmd_decide(args) -> int:
    from .contracts import load_jsonl, validate_request_payload
    from .export import load_exported_model

    raw = json.loads(Path(args.request).read_text(encoding="utf-8")) if args.request else json.loads(sys.stdin.read())
    case = validate_request_payload(raw)
    model, policy = load_exported_model(Path(args.model_dir))
    threshold = args.threshold if args.threshold is not None else policy["threshold"]
    probs = model.score([case])[0]
    from .policy import apply_policy

    response = apply_policy(
        case,
        probs,
        threshold=threshold,
        model_revision=model.model_revision,
        calibration_revision=policy.get("calibration_revision", ""),
    )
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_check(args) -> int:
    from .contracts import DecisionCase, load_jsonl

    path = Path(args.file)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    n = 0
    for row in load_jsonl(str(path)):
        DecisionCase.from_dict(row)
        n += 1
    print(f"ok: {n} valid cases in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
