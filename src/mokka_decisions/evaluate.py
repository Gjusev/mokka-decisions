"""Metrics and the common evaluation runner.

Definitions (fixed here, used everywhere):

* accuracy — fraction of accepted decisions (policy applied) whose choice
  equals the target; reported alongside coverage so the pair is readable;
* forced accuracy / macro-F1 — no abstention, argmax vs target;
* NLL / Brier — over the calibrated probability of the gold option, computed
  on finite rows only (errors excluded, counted separately);
* ECE — 15 equal-width bins over max-probability, weighted by bin size;
* bootstrap CIs — resampled by group_id (paraphrases/translations of one case
  are not independent observations).

Latency is measured by the backend itself (batch=1, per case, wall clock) and
summarised as p50/p95; it includes tokenisation and scoring for encoder
backends, and whatever the backend does for baselines.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .backends.base import Backend, BackendRow
from .contracts import NONE_ID, DecisionCase
from .policy import apply_policy

ECE_BINS = 15


def macro_f1(rows: Sequence[BackendRow], cases: Sequence[DecisionCase]) -> float:
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    for row, case in zip(rows, cases):
        gold = case.target_option
        pred = row.candidate if row.candidate else "__error__"
        if pred == gold:
            per_class[gold][0] += 1
        else:
            per_class[pred][1] += 1
            per_class[gold][2] += 1
    f1s = []
    for cls, (tp, fp, fn) in per_class.items():
        if cls == "__error__":
            continue
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def nll_brier(rows: Sequence[BackendRow], cases: Sequence[DecisionCase]) -> dict:
    """NLL of the gold option and two Brier variants (metrics v2).

    ``brier`` is the full multiclass Brier score: the sum over ALL options of
    (p_i - 1[i == gold])^2 — range [0, 2]. ``brier_gold`` is the legacy v1
    value (1 - p_gold)^2 kept for continuity with earlier reports; historical
    numbers stay comparable under that key only.
    """
    nlls, briers, briers_gold, skipped = [], [], [], 0
    for row, case in zip(rows, cases):
        p = row.probabilities.get(case.target_option)
        if p is None or not math.isfinite(p):
            skipped += 1
            continue
        p = min(max(p, 1e-12), 1 - 1e-12)
        nlls.append(-math.log(p))
        briers_gold.append((1.0 - p) ** 2)
        total = (1.0 - p) ** 2
        for oid, pi in row.probabilities.items():
            if oid != case.target_option:
                total += pi**2
        briers.append(total)
    return {
        "nll": sum(nlls) / len(nlls) if nlls else None,
        "brier": sum(briers) / len(briers) if briers else None,
        "brier_gold": sum(briers_gold) / len(briers_gold) if briers_gold else None,
        "probability_rows": len(nlls),
        "rows_without_gold_probability": skipped,
        "metrics_version": 2,
    }


def ece_with_cases(
    rows: Sequence[BackendRow], cases: Sequence[DecisionCase], bins: int = ECE_BINS
) -> float | None:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row, case in zip(rows, cases):
        if not row.probabilities:
            continue
        conf = max(row.probabilities.values())
        correct = 1 if row.candidate == case.target_option else 0
        idx = min(int(conf * bins), bins - 1)
        buckets[idx].append((conf, correct))
    n = sum(len(b) for b in buckets)
    if n == 0:
        return None
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        size = len(bucket)
        avg_conf = sum(c for c, _ in bucket) / size
        avg_acc = sum(a for _, a in bucket) / size
        ece += size / n * abs(avg_acc - avg_conf)
    return ece


def summarize(
    rows: Sequence[BackendRow],
    cases: Sequence[DecisionCase],
    *,
    threshold: float | None = None,
    languages: Sequence[str] | None = None,
) -> dict:
    """One summary dict; with a threshold the selective metrics join in."""
    ok = [r for r in rows if not r.error]
    errors = len(rows) - len(ok)
    forced_correct = sum(
        1 for r, c in zip(rows, cases) if r.candidate and r.candidate == c.target_option
    )
    out = {
        "n_cases": len(cases),
        "backend_errors": errors,
        "forced_accuracy": forced_correct / len(cases) if cases else None,
        "macro_f1": macro_f1(rows, cases),
        **nll_brier(rows, cases),
        "ece": ece_with_cases(rows, cases),
        "latency_ms_p50": _percentile([r.latency_ms for r in ok], 50),
        "latency_ms_p95": _percentile([r.latency_ms for r in ok], 95),
    }
    if threshold is not None:
        accepted = 0
        accepted_errors = 0
        none_accepted = 0
        for r, c in zip(rows, cases):
            if r.error or not r.probabilities:
                continue
            resp = apply_policy(
                c, r.probabilities, threshold=threshold, languages=languages
            )
            if resp.abstain:
                continue
            accepted += 1
            if resp.choice != c.target_option:
                accepted_errors += 1
            if resp.choice == NONE_ID:
                none_accepted += 1
        out.update(
            {
                "threshold": threshold,
                "accepted": accepted,
                "coverage": accepted / len(cases) if cases else 0.0,
                "selective_risk": accepted_errors / accepted if accepted else None,
                "none_accepted": none_accepted,
            }
        )
    return out


def summarize_by(
    rows: Sequence[BackendRow], cases: Sequence[DecisionCase], key: str, **kwargs
) -> dict[str, dict]:
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for row, case in zip(rows, cases):
        groups[getattr(case, key)][0].append(row)
        groups[getattr(case, key)][1].append(case)
    return {k: summarize(rs, cs, **kwargs) for k, (rs, cs) in sorted(groups.items())}


def bootstrap_accuracy_ci(
    rows: Sequence[BackendRow],
    cases: Sequence[DecisionCase],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile CI for forced accuracy, resampling whole group_ids."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, case in enumerate(cases):
        by_group[case.group_id].append(i)
    groups = list(by_group.values())
    correct = [
        1 if rows[i].candidate and rows[i].candidate == cases[i].target_option else 0
        for i in range(len(cases))
    ]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        picks = []
        for _ in range(len(groups)):
            picks.extend(rng.choice(groups))
        stats.append(sum(correct[i] for i in picks) / len(picks))
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    point = sum(correct) / len(correct)
    return {"accuracy": point, "ci95": [lo, hi], "n_boot": n_boot}


def oos_false_acceptance(
    rows: Sequence[BackendRow],
    cases: Sequence[DecisionCase],
    *,
    threshold: float,
) -> dict:
    """On gold=none cases: accepted non-none decisions are false acceptances."""
    total = accepted = false_accepted = 0
    for row, case in zip(rows, cases):
        if case.target_option != NONE_ID:
            continue
        total += 1
        if row.error or not row.probabilities:
            continue
        candidate = max(row.probabilities, key=row.probabilities.get)
        if row.probabilities[candidate] >= threshold and candidate != NONE_ID:
            accepted += 1
            false_accepted += 1
        elif row.probabilities[candidate] >= threshold:
            accepted += 1
    return {
        "oos_cases": total,
        "oos_accepted_any": accepted,
        "oos_false_accepted": false_accepted,
        "false_acceptance_rate": false_accepted / total if total else None,
    }


def permutation_flip_rate(base_rows: Sequence[BackendRow], perm_rows: Sequence[BackendRow]) -> dict:
    flips = 0
    pairs = 0
    for a, b in zip(base_rows, perm_rows):
        if a.error or b.error:
            continue
        pairs += 1
        if a.candidate != b.candidate:
            flips += 1
    return {"pairs": pairs, "flips": flips, "flip_rate": flips / pairs if pairs else None}


def run_backend(
    backend: Backend, cases: Sequence[DecisionCase], *, chunk: int = 256
) -> list[BackendRow]:
    """Run a backend over cases in bounded chunks (peak-memory guard)."""
    rows: list[BackendRow] = []
    for start in range(0, len(cases), chunk):
        rows.extend(backend.decide(cases[start : start + chunk]))
    return rows


def dump_rows(rows: Sequence[BackendRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(int(q / 100 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]
