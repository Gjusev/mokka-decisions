"""Common backend interface: every comparator emits one row per case.

A row records the prediction *and* how long it took *and* whether it failed.
Failures stay in the file with ``error`` set — a backend that cannot score a
case is a measured fact, never a silently dropped denominator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

from ..contracts import DecisionCase


@dataclass
class BackendRow:
    case_id: str
    group_id: str
    source: str
    language: str
    domain: str
    backend: str
    model_revision: str
    candidate: str  # "" on error
    probabilities: dict[str, float]  # {} on error
    latency_ms: float
    error: str = ""
    # filled by the policy, not the backend
    choice: str | None = None
    abstain: bool = False
    reason_code: str = ""
    # optional backend-native scores before any renormalisation (e.g. GLiNER's
    # independent sigmoid confidences); kept for separate analysis
    raw_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_probabilities(probs: Mapping[str, float]) -> dict[str, float]:
    """Clamp, drop negatives/NaN and renormalise to sum 1 within 1e-6."""
    clean = {}
    for k, v in probs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        v = float(v)
        if math.isnan(v) or v < 0:
            continue
        clean[k] = min(v, 1.0)
    total = math.fsum(clean.values())
    if total <= 0:
        n = len(clean) or 1
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


class Backend:
    """Subclass and implement ``decide``; timing and error capture are shared."""

    name = "backend"
    model_revision = ""

    def decide(self, cases: Sequence[DecisionCase]) -> list[BackendRow]:
        rows: list[BackendRow] = []
        for case in cases:
            start = time.perf_counter()
            try:
                raw = self.score_one(case)
                probs = normalise_probabilities(raw)
                latency = (time.perf_counter() - start) * 1000.0
                candidate = max(probs, key=probs.get) if probs else ""
                rows.append(
                    BackendRow(
                        case_id=case.id,
                        group_id=case.group_id,
                        source=case.source,
                        language=case.language,
                        domain=case.domain,
                        backend=self.name,
                        model_revision=self.model_revision,
                        candidate=candidate,
                        probabilities=probs,
                        latency_ms=latency,
                        raw_scores=dict(raw) if raw != probs else None,
                    )
                )
            except Exception as exc:  # measured failure, never dropped
                latency = (time.perf_counter() - start) * 1000.0
                rows.append(
                    BackendRow(
                        case_id=case.id,
                        group_id=case.group_id,
                        source=case.source,
                        language=case.language,
                        domain=case.domain,
                        backend=self.name,
                        model_revision=self.model_revision,
                        candidate="",
                        probabilities={},
                        latency_ms=latency,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return rows

    def score_one(self, case: DecisionCase) -> dict[str, float]:
        raise NotImplementedError
