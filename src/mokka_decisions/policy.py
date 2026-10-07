"""Abstention policy: when a candidate becomes an accepted choice.

Two different outcomes, never conflated:

* choosing the ``none`` option is an *explicit decision* ("no listed option
  applies") and is accepted like any other high-confidence answer;
* *abstaining* means confidence is below the validated threshold, the input is
  invalid/unsupported, or scope validation fails — ``choice`` stays null.

The threshold is picked on ``cal_policy`` to hit a target selective risk
(error rate among accepted decisions) at maximum coverage. ``none``-decisions
count as correct when the gold is none and as errors otherwise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from .contracts import NONE_ID, DecisionCase, DecisionResponse


def apply_policy(
    case: DecisionCase,
    probabilities: dict[str, float],
    *,
    threshold: float,
    model_revision: str = "",
    calibration_revision: str = "",
    max_state_chars: int = 20_000,
    max_options: int = 128,
    languages: Sequence[str] | None = None,
) -> DecisionResponse:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    candidate = max(probabilities, key=probabilities.get)
    confidence = probabilities[candidate]

    reason = "accepted"
    if len(case.state) > max_state_chars:
        reason = "unsupported_length"
    elif len(case.options) > max_options:
        reason = "invalid_input"
    elif languages is not None and case.language not in languages:
        reason = "language_not_validated"
    elif confidence < threshold:
        reason = "below_validated_threshold"

    abstain = reason != "accepted"
    return DecisionResponse(
        candidate=candidate,
        probabilities=probabilities,
        choice=None if abstain else candidate,
        confidence=confidence,
        abstain=abstain,
        reason_code=reason,
        model_revision=model_revision,
        calibration_revision=calibration_revision,
    )


def pick_threshold(
    cases: Sequence[DecisionCase],
    prob_rows: Sequence[dict[str, float]],
    *,
    target_risk: float = 0.05,
    grid: Sequence[float] | None = None,
) -> dict:
    """Smallest threshold whose accepted-decision error <= target_risk.

    ``none``-gold cases count as decisions: accepting an OOS case as a real
    option is an error; accepting it as ``none`` is correct. If even total
    abstention cannot reach the target (accepted set too small), the largest
    threshold is returned with ``met_target=False``.
    """
    if len(cases) != len(prob_rows):
        raise ValueError("cases and probability rows must align")
    if grid is None:
        grid = [round(0.30 + 0.005 * i, 4) for i in range(int((0.995 - 0.30) / 0.005) + 1)]

    def evaluate(threshold: float) -> dict:
        accepted = errors = 0
        for case, probs in zip(cases, prob_rows):
            if not probs:
                continue
            candidate = max(probs, key=probs.get)
            if probs[candidate] < threshold:
                continue
            accepted += 1
            if candidate != case.target_option:
                errors += 1
        risk = errors / accepted if accepted else float("nan")
        return {
            "threshold": threshold,
            "accepted": accepted,
            "coverage": accepted / len(cases) if cases else 0.0,
            "risk": risk,
        }

    best = None
    for threshold in grid:
        stats = evaluate(threshold)
        if stats["accepted"] == 0:
            continue
        if stats["risk"] <= target_risk:
            best = stats  # grid ascends -> first hit is max coverage
            break
        best = stats  # keep the safest (largest threshold) so far
    if best is None:
        best = evaluate(1.01)  # abstain on everything
        best["threshold"] = 1.01
    best["met_target"] = bool(
        best["accepted"] > 0 and not math.isnan(best["risk"]) and best["risk"] <= target_risk
    )
    return best


def risk_coverage_curve(
    cases: Sequence[DecisionCase],
    prob_rows: Sequence[dict[str, float]],
    grid: Sequence[float] | None = None,
) -> list[dict]:
    if grid is None:
        grid = [round(0.05 * i, 3) for i in range(20)]
    out = []
    for threshold in grid:
        accepted = errors = 0
        for case, probs in zip(cases, prob_rows):
            if not probs:
                continue
            candidate = max(probs, key=probs.get)
            if probs[candidate] < threshold:
                continue
            accepted += 1
            if candidate != case.target_option:
                errors += 1
        out.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "coverage": accepted / len(cases) if cases else 0.0,
                "risk": (errors / accepted) if accepted else None,
            }
        )
    return out


def save_policy(path: str | Path, policy: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")


def load_policy(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
