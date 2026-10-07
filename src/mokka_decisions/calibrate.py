"""Temperature scaling on a dedicated split (Guo et al., ICML 2017).

A single positive temperature over decision softmaxes: preserves the argmax,
improves the probability *quality* (NLL/Brier/ECE). Fitted on
``cal_temperature`` — never on dev or test.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from .contracts import DecisionCase


def _nll(logits: Sequence[float], target_index: int, temperature: float) -> float:
    scaled = [l / temperature for l in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return -(scaled[target_index] - m - math.log(total))


def fit_temperature(
    cases: Sequence[DecisionCase],
    logit_rows: Sequence[dict[str, float]],
    *,
    max_iter: int = 100,
) -> float:
    """Golden-section search for the T minimising NLL (T>0, argmax-preserving)."""
    if len(cases) != len(logit_rows):
        raise ValueError("cases and logit rows must align")

    def total_nll(t: float) -> float:
        return math.fsum(
            _nll(
                [logit_rows[i][oid] for oid in case.option_ids],
                case.option_ids.index(case.target_option),
                t,
            )
            for i, case in enumerate(cases)
            if case.target_option in case.option_ids
        )

    lo, hi = 0.05, 20.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(max_iter):
        if total_nll(c) < total_nll(d):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
        if b - a < 1e-6:
            break
    t = (a + b) / 2.0
    return round(t, 6)


def calibration_revision(temperature: float, case_ids: Sequence[str]) -> str:
    payload = json.dumps(
        {"t": temperature, "n": len(case_ids), "ids": sorted(case_ids)[:1000]}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def save_calibration(path: str | Path, temperature: float, revision: str, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"temperature": temperature, "revision": revision, "stats": stats}, indent=2),
        encoding="utf-8",
    )


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
