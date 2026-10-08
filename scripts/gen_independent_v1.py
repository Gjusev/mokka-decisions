"""Generate the FROZEN independent typed-decisions evaluation set (indep-v1).

Materialises docs/typed-independent-eval-protocol.md (source #1: own generator,
rule-verifiable labels). Three NOVEL families — shipment routing, sensor
monitoring, subscription billing — disjoint from every upstream domain. Every
gold label is computed from the state by an explicit rule; rubrics for ordinal
(score) questions are fully stated in the option texts so answers are deducible
from the supplied information alone.

Freeze contract: pinned seed, deterministic templates, option order shuffled
per question by a stable hash (choice only — noul keeps the upstream
false/true order and score levels are ordered by definition). The generator's
sha256 + this seed + the output manifest freeze the set BEFORE any model runs
it. Nothing here derives from the historical test, its errors or predictions.

Checks enforced at build time: unique correct choice, no label-leak tokens,
uniform gold positions for choice, and empty state-text overlap (sha256) with
every consulted upstream split (train + the four domain tests).

Output: data/independent-v1/cases.jsonl (upstream-style rows) + manifest.json.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008
N_CASES_PER_FAMILY = 50
FORBIDDEN_TOKENS = ("correct", "answer", "gold", "label", "solution")


def stable_shuffle(seq, key: str):
    order = sorted(range(len(seq)), key=lambda i: hashlib.sha256(f"{key}|{i}".encode()).hexdigest())
    return [seq[i] for i in order]


# ---------------------------------------------------------------- family A
def gen_shipment(rng: random.Random) -> tuple[dict, list[dict]]:
    weight = round(rng.uniform(1.5, 40.0), 1)
    in_union = rng.random() < 0.5
    dest = rng.choice(["FR", "NL", "AT"]) if in_union else rng.choice(["CH", "NO", "GB"])
    items = [
        {"name": n, "qty": rng.randint(1, 6), "unit_price_eur": round(rng.uniform(8, 900), 2)}
        for n in rng.sample(["cable reel", "bearing set", "filter cartridge", "valve body",
                             "gasket kit", "mounting plate", "sensor probe", "hose assembly"], 3)
    ]
    value = round(sum(i["qty"] * i["unit_price_eur"] for i in items), 2)
    deadline = rng.choice([24, 48, 72])
    names = rng.sample(["AlphaFreight", "BetaExpress", "CargoJet", "DeltaLine", "EcoShip"], 4)
    # exactly one cheapest carrier qualifies (deadline + weight); others are
    # either slower than the deadline, too weak, or strictly more expensive
    winner_cost = round(rng.uniform(20, 60), 2)
    carriers = [{"name": names[0], "cost_eur": winner_cost, "max_kg": round(weight + rng.uniform(5, 20), 1),
                 "transit_hours": max(6, deadline - rng.randint(4, 12))}]
    for n in names[1:]:
        if rng.random() < 0.5:  # fails a constraint: cheaper but too slow or too weak
            fails_weight = rng.random() < 0.5
            carriers.append({"name": n, "cost_eur": round(rng.uniform(5, 18), 2),
                             "max_kg": round(weight - rng.uniform(0.5, 5), 1) if fails_weight
                             else round(weight + 5, 1),
                             "transit_hours": deadline + rng.randint(2, 30) if not fails_weight
                             else max(6, deadline - rng.randint(0, 10))})
        else:  # qualifies but strictly pricier
            carriers.append({"name": n, "cost_eur": round(winner_cost + rng.uniform(4, 40), 2),
                             "max_kg": round(weight + rng.uniform(5, 25), 1),
                             "transit_hours": max(6, deadline - rng.randint(0, 10))})
    # repair accidental winners among non-target carriers
    for c in carriers[1:]:
        qualifies = c["transit_hours"] <= deadline and c["max_kg"] >= weight
        if qualifies and c["cost_eur"] <= winner_cost:
            c["cost_eur"] = round(winner_cost + rng.uniform(4, 40), 2)
    state = {
        "order_id": f"ORD-{rng.randint(10000, 99999)}",
        "origin_country": "DE", "destination_country": dest,
        "customs_union_with_origin": in_union,
        "weight_kg": weight, "items": items, "deadline_hours": deadline, "carriers": carriers,
    }
    gold_carrier = min((c for c in carriers if c["transit_hours"] <= deadline and c["max_kg"] >= weight),
                       key=lambda c: c["cost_eur"])["name"]
    tiers = ["light: up to 5.0 kg total and no fragile items",
             "protective: any fragile item regardless of weight",
             "heavy: more than 20.0 kg total and no fragile items",
             "reinforced: fragile items and more than 20.0 kg total"]
    fragile = rng.random() < 0.3
    state["fragile_items"] = fragile
    if fragile and weight > 20:
        packaging = "reinforced"
    elif fragile:
        packaging = "protective"
    elif weight > 20:
        packaging = "heavy"
    else:
        packaging = "light"
    questions = [
        {"qid": "q_carrier", "type": "choice",
         "instructions": "Which carrier should handle this order? Select the cheapest carrier whose transit time meets the delivery deadline and whose weight cap covers the shipment weight.",
         "criteria": {c["name"]: c["name"] for c in carriers}, "gold": gold_carrier},
        {"qid": "q_customs", "type": "noul",
         "instructions": "A customs declaration is required only when origin and destination countries are in different customs unions. Is a customs declaration required for this shipment?",
         "criteria": None, "gold": "false" if in_union else "true"},
        {"qid": "q_priority", "type": "score",
         "instructions": "Rate the picking priority of this order based on its total item value.",
         "criteria": ["routine: total value below 250.00 EUR",
                      "standard: total value from 250.00 EUR to 999.99 EUR",
                      "high: total value from 1000.00 EUR to 2499.99 EUR",
                      "critical: total value of 2500.00 EUR or more"],
         "gold": str(min(3, int(value // 1000) + (1 if value >= 250 else 0)))},
        {"qid": "q_insurance", "type": "noul",
         "instructions": "The base contract includes insurance up to 1000.00 EUR of declared value. Does this order exceed that included insurance cover?",
         "criteria": None, "gold": "true" if value > 1000 else "false"},
        {"qid": "q_packaging", "type": "choice",
         "instructions": "Which packaging tier applies to this shipment?",
         "criteria": {t.split(":")[0]: t for t in tiers}, "gold": packaging},
    ]
    # fix score gold from the rubric thresholds exactly as written
    v = value
    questions[2]["gold"] = "0" if v < 250 else "1" if v < 1000 else "2" if v < 2500 else "3"
    return state, questions


# ---------------------------------------------------------------- family B
def gen_sensor(rng: random.Random) -> tuple[dict, list[dict]]:
    metrics = ["temperature_c", "pressure_bar", "vibration_mm_s"]
    limits = {"temperature_c": {"warn": 80, "alarm": 90},
              "pressure_bar": {"warn": 6.0, "alarm": 7.0},
              "vibration_mm_s": {"warn": 8.0, "alarm": 12.0}}
    breach = rng.choice(metrics)
    readings, ratios = {}, {}
    for m in metrics:
        lim = limits[m]
        if m == breach:
            readings[m] = round(lim["alarm"] * rng.uniform(1.02, 1.6), 2)
        else:
            readings[m] = round(lim["warn"] * rng.uniform(0.55, 0.98), 2)
        ratios[m] = readings[m] / lim["alarm"]
    hours = rng.randint(1500, 6000)
    interval = rng.choice([2000, 4000, 6000])
    calib = rng.randint(50, 400)
    state = {
        "device_id": f"PMP-{rng.randint(1000, 9999)}", "site": f"Pump station {rng.randint(2, 19)}",
        "hours_since_service": hours, "service_interval_hours": interval,
        "readings": readings, "limits": limits,
        "last_calibration_days_ago": calib,
    }
    max_ratio = max(ratios.values())
    steps = ["log_only: no alarm threshold crossed",
             "notify_operator: an alarm threshold crossed but every reading is below 1.5 times its alarm threshold",
             "reduce_load: any reading at or above 1.5 times its alarm threshold and below 2.0 times",
             "shutdown: any reading at or above 2.0 times its alarm threshold"]
    step = "log_only" if max_ratio < 1 else "notify_operator" if max_ratio < 1.5 else "reduce_load" if max_ratio < 2 else "shutdown"
    n_warn = sum(1 for m in metrics if readings[m] >= limits[m]["warn"])
    sev = ["routine: no reading at or above its warning threshold",
           "watch: exactly one reading at or above its warning threshold",
           "elevated: exactly two readings at or above their warning thresholds",
           "urgent: all three readings at or above their warning thresholds"]
    sev_gold = str(min(3, n_warn))
    questions = [
        {"qid": "q_breach", "type": "choice",
         "instructions": "Which reading is above its alarm threshold? Limits are listed in the state.",
         "criteria": {m: m for m in metrics}, "gold": breach},
        {"qid": "q_service", "type": "noul",
         "instructions": "Is preventive maintenance overdue for this device according to its service interval?",
         "criteria": None, "gold": "true" if hours >= interval else "false"},
        {"qid": "q_severity", "type": "score",
         "instructions": "Rate the maintenance urgency of this device.",
         "criteria": sev, "gold": sev_gold},
        {"qid": "q_calibration", "type": "noul",
         "instructions": "The annual calibration check repeats every 365 days. Is the next calibration due within the next 30 days?",
         "criteria": None, "gold": "true" if calib >= 335 else "false"},
        {"qid": "q_action", "type": "choice",
         "instructions": "Which response step does the escalation policy require for the current readings?",
         "criteria": {s.split(":")[0]: s for s in steps}, "gold": step.split(":")[0]},
    ]
    return state, questions


# ---------------------------------------------------------------- family C
def gen_billing(rng: random.Random) -> tuple[dict, list[dict]]:
    included = rng.choice([500, 1000, 2000])
    usage = rng.randint(100, int(included * 1.4))
    overdue = rng.randint(1, 45)
    plan_change = None
    if rng.random() < 0.7:
        plan_change = {"to_plan": rng.choice(["business", "enterprise"]),
                       "effective_day": rng.randint(2, 30)}
    first_overage = rng.random() < 0.5
    state = {
        "account_id": f"ACC-{rng.randint(10000, 99999)}",
        "plan": "pro", "plan_price_eur": 29.0,
        "plan_change": plan_change,
        "billing_period_days": 30,
        "payment_status": "overdue", "overdue_days": overdue,
        "usage_units": usage, "included_units": included,
        "first_period_with_overage": first_overage,
    }
    dun = ["reminder: 1 to 7 days overdue",
           "formal_notice: 8 to 15 days overdue",
           "suspension_warning: 16 to 30 days overdue",
           "suspension: more than 30 days overdue"]
    dun_gold = "0" if overdue <= 7 else "1" if overdue <= 15 else "2" if overdue <= 30 else "3"
    util = usage / included
    risk = ["low: usage below 50 percent of included units",
            "moderate: usage from 50 to 79 percent of included units",
            "high: usage from 80 to 99 percent of included units",
            "saturation: usage at or above included units"]
    risk_gold = "0" if util < 0.5 else "1" if util < 0.8 else "2" if util < 1.0 else "3"
    overage = usage - included
    if overage <= 0:
        credit = "none"
    elif overage < 50 and first_overage:
        credit = "grace"
    else:
        credit = "standard"
    questions = [
        {"qid": "q_dunning", "type": "score",
         "instructions": "Which dunning stage applies to this account?",
         "criteria": dun, "gold": dun_gold},
        {"qid": "q_overage", "type": "noul",
         "instructions": "Does this account exceed its included monthly usage units?",
         "criteria": None, "gold": "true" if usage > included else "false"},
        {"qid": "q_risk", "type": "score",
         "instructions": "Rate the plan-saturation risk of this account relative to its included units.",
         "criteria": risk, "gold": risk_gold},
        {"qid": "q_proration", "type": "noul",
         "instructions": "A mid-cycle plan change that takes effect after day 1 of the billing cycle requires a prorated charge on the next invoice. Does this account require a prorated charge?",
         "criteria": None, "gold": "true" if plan_change else "false"},
        {"qid": "q_credit", "type": "choice",
         "instructions": "Which overage credit applies to this account under the first-occurrence grace policy?",
         "criteria": {"none": "none: usage at or below included units",
                      "grace": "grace: overage below 50 units and this is the first period with overage",
                      "standard": "standard: overage of 50 units or more, or a repeated period with overage"},
         "gold": credit},
    ]
    return state, questions


FAMILIES = {"shipment_routing": gen_shipment, "sensor_monitoring": gen_sensor, "subscription_billing": gen_billing}


def main() -> int:
    out_dir = ROOT / "data/independent-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_state_hashes = set()
    # overlap guard: every consulted upstream split (train + 4 domain tests)
    import pyarrow.parquet as pq

    for pq_path in [ROOT / "data/typed-v1/train-00000-of-00001.parquet"] + [
        ROOT / ".cache/typed-holdout-probe" / f"{c}-test.parquet"
        for c in ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
    ]:
        if pq_path.exists():
            for case in pq.read_table(pq_path).to_pylist():
                all_state_hashes.add(hashlib.sha256(case["state"].encode()).hexdigest())

    rows = []
    pos_hist = collections.Counter()
    type_counts = collections.Counter()
    for fam_idx, (fam, gen) in enumerate(sorted(FAMILIES.items())):
        rng = random.Random(SEED + fam_idx * 1000)
        for i in range(N_CASES_PER_FAMILY):
            state, questions = gen(rng)
            state_json = json.dumps(state, ensure_ascii=False)
            assert hashlib.sha256(state_json.encode()).hexdigest() not in all_state_hashes, "state overlap with upstream"
            blob = state_json + " ".join(q["instructions"] for q in questions)
            for tok in FORBIDDEN_TOKENS:
                assert tok not in blob.lower(), f"leak token {tok!r} in {fam} case {i}"
            qdict, gold = {}, {}
            for q in questions:
                criteria = q["criteria"]
                if q["type"] == "choice":
                    items = stable_shuffle(list(criteria.items()), f"{fam}|{i}|{q['qid']}")
                    criteria = dict(items)
                    keys = list(criteria)
                    assert keys.count(q["gold"]) == 1
                    gold[q["qid"]] = str(keys.index(q["gold"]))
                elif q["type"] == "score":
                    keys = [str(x) for x in range(len(criteria))]
                    gold[q["qid"]] = str(keys.index(q["gold"]))
                else:
                    keys = ["false", "true"]
                    gold[q["qid"]] = str(keys.index(q["gold"]))
                qdict[q["qid"]] = {"instructions": q["instructions"], "type": q["type"], "criteria": criteria}
                type_counts[q["type"]] += 1
                if q["type"] in ("choice", "noul"):
                    pos_hist[(q["type"], gold[q["qid"]])] += 1
            rows.append({"id": f"indep-v1|{fam}|{i:03d}", "family": fam,
                         "state": state_json, "questions": qdict, "gold": gold})
    # position-uniformity sanity (no positional leak): max share per position <= 0.65
    for t in ("choice", "noul"):
        total = sum(v for (tt, _), v in pos_hist.items() if tt == t)
        positions = sorted({k for (tt, k) in pos_hist if tt == t})
        for k in positions:
            share = pos_hist[(t, k)] / total
            assert share <= 0.65, f"positional leak: {t} gold={k} share={share:.2f}"

    path = out_dir / "cases.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    gen_src = Path(__file__).read_bytes()
    manifest = {
        "set": "indep-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/gen_independent_v1.py", "generator_sha256": hashlib.sha256(gen_src).hexdigest(),
        "seed": SEED, "cases": len(rows), "questions": sum(type_counts.values()),
        "per_family": dict(collections.Counter(r["family"] for r in rows)),
        "per_type": dict(type_counts),
        "gold_position_histogram": {f"{t}|{k}": v for (t, k), v in sorted(pos_hist.items())},
        "checks": {
            "state_overlap_with_consulted_splits": "empty (sha256 of state text vs train + 4 domain tests)",
            "choice_unique_correct": "asserted at build",
            "leak_tokens": list(FORBIDDEN_TOKENS),
            "choice_noul_position_max_share": 0.65,
            "labels": "rule-computed by the generator; rubrics fully stated in score/choice criteria",
        },
        "scope": "generalisation to NOVEL families with rule-certain labels; one-hot gold by construction; "
                 "NOT a sample of the upstream distribution — declared as such in any comparison",
        "protocol": "docs/typed-independent-eval-protocol.md",
        "files": {"cases.jsonl": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "bytes": path.stat().st_size}},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("cases", "questions", "per_family", "per_type",
                                               "gold_position_histogram")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
