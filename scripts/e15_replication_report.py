"""Paired matched-seed report for E15 (control vs dataset-native candidate).

Commission 2026-10-08 step 3. Seeds 42/43/44 are the DECIDING matched pairs;
seed 45 (candidate-only) is exploratory and never selects anything. Fails
explicitly on missing results, missing checkpoints or id mismatch.

Inputs (local collections):
  runs/kaggle-typed-e15/         v1: e15_control/, e15_probs/ + e15_results.json (seed 42)
  runs/kaggle-typed-e15-seeds/   v2: e15_{control,probs}_s{43,44}/, e15_probs_s45/
                                 + e15_s{43,44,45}_results.json

Per arm: dev per-question inference from checkpoint_best (CPU, frozen protocol
T=1.0 max_length 512 batch 16), cross-checked against the kernel-reported
best_dev_accuracy. Grouped bootstrap (resampling original cases) for the
paired accuracy delta per seed and pooled.

Output: runs/typed-decisions/e15_replication_report.json (+ markdown to stdout).
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

V1 = ROOT / "runs/kaggle-typed-e15"
V2 = ROOT / "runs/kaggle-typed-e15-seeds"


def _kernel_dev(ckpt_dir: Path, key: str | None, seed: int) -> float | None:
    """Kernel-reported best dev: per-seed stage JSON when present, else the
    arm's own run_report.json (both were written by the kernel session)."""
    sj = V2 / f"e15_s{seed}_results.json"
    if sj.exists() and key:
        try:
            return json.loads(sj.read_text(encoding="utf-8"))[key]["best_dev_accuracy"]
        except (KeyError, json.JSONDecodeError):
            pass
    rr = ckpt_dir / "run_report.json"
    if rr.exists():
        return json.loads(rr.read_text(encoding="utf-8"))["best_dev_accuracy"]
    return None


ARMS = {  # recipe -> {seed: checkpoint_dir}
    "control": {42: V1 / "e15_control", 43: V2 / "e15_control_s43", 44: V2 / "e15_control_s44"},
    "candidate": {42: V1 / "e15_probs", 43: V2 / "e15_probs_s43",
                  44: V2 / "e15_probs_s44", 45: V2 / "e15_probs_s45"},
}
DECIDING_SEEDS = (42, 43, 44)
EXPLORATORY = {45}


def load_dev():
    cases = [json.loads(l) for l in (ROOT / "data/typed-v1/cases/dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    probs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/typed-v1/cases/probabilities_dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    assert set(probs) == {c["id"] for c in cases}, "dev id mismatch"
    return cases, probs


def infer_arm(ckpt_dir: Path):
    import torch
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import OptionScorer, ScorerInference, load_checkpoint
    from transformers import AutoTokenizer

    torch.set_num_threads(8)
    m = OptionScorer("jhu-clsp/mmBERT-base")
    load_checkpoint(m, ckpt_dir / "checkpoint_best.safetensors")
    sc = ScorerInference(m, AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"),
                         device="cpu", temperature=1.0, max_length=512, batch_decisions=16)
    cases_raw, gold_probs = load_dev()
    cases = [DecisionCase(
        id=c["id"], group_id=c["id"].rsplit(":", 1)[0], source="typed-dev", language="en",
        domain="benchmark", state=c["state"], question=c["question"],
        options=tuple(Option(id=f"o{i}", description=o["description"]) for i, o in enumerate(c["options"])),
        target_kind="single", target_option=c["target_option"],
    ) for c in cases_raw]
    rows = []
    for s in range(0, len(cases), 16):
        for case, p_ in zip(cases[s:s+16], sc.score(cases[s:s+16])):
            gold_idx = case.option_ids.index(case.target_option)
            pv = [p_[oid] for oid in case.option_ids]
            vec = gold_probs[case.id]["probabilities"]
            qtype = gold_probs[case.id]["question_type"]
            eps = 1e-12
            rec = {
                "id": case.id, "group": case.group_id, "type": qtype,
                "correct": int(max(p_, key=p_.get) == case.target_option),
                "nll": -math.log(max(pv[gold_idx], eps)),
                "kl": sum(v * (math.log(max(v, eps)) - math.log(max(pi, eps))) for v, pi in zip(vec, pv) if v > 0),
                "brier": sum((pi - (1.0 if i == gold_idx else 0.0)) ** 2 for i, pi in enumerate(pv)),
            }
            if qtype == "score":
                rec["mae"] = abs(sum(i * pi for i, pi in enumerate(pv)) - gold_idx)
            rows.append(rec)
    return rows


def summarize(rows):
    by_type = defaultdict(lambda: dict(n=0, ok=0))
    for r in rows:
        by_type[r["type"]]["n"] += 1
        by_type[r["type"]]["ok"] += r["correct"]
    mae = [r["mae"] for r in rows if "mae" in r]
    return {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "by_type": {t: v["ok"] / v["n"] for t, v in by_type.items()},
        "nll": sum(r["nll"] for r in rows) / len(rows),
        "kl": sum(r["kl"] for r in rows) / len(rows),
        "brier": sum(r["brier"] for r in rows) / len(rows),
        "mae": sum(mae) / len(mae) if mae else None,
        "n": len(rows),
    }


def grouped_bootstrap_delta(rows_a, rows_b, n_boot=20000, seed=20261008):
    """CI for accuracy(a) - accuracy(b), resampling original cases (groups)."""
    groups = defaultdict(list)
    for ra, rb in zip(rows_a, rows_b):
        assert ra["id"] == rb["id"]
        groups[ra["group"]].append((ra["correct"], rb["correct"]))
    gids = sorted(groups)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [groups[rng.choice(gids)] for _ in range(len(gids))]
        a = sum(x[0] for g in sample for x in g)
        b = sum(x[1] for g in sample for x in g)
        n = sum(len(g) for g in sample)
        diffs.append((a - b) / n)
    diffs.sort()
    point = sum(r["correct"] for r in rows_a) / len(rows_a) - sum(r["correct"] for r in rows_b) / len(rows_b)
    return {"point": point, "ci95": [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]]}


def main() -> int:
    missing = []
    for recipe, seeds in ARMS.items():
        for seed, ckpt_dir in seeds.items():
            if not (ckpt_dir / "checkpoint_best.safetensors").exists():
                missing.append(f"{recipe} s{seed}: checkpoint {ckpt_dir}")
    if missing:
        raise SystemExit("MISSING INPUTS (no invention, fail explicit):\n  " + "\n  ".join(missing))

    data = {}  # (recipe, seed) -> rows
    kernel_dev = {}
    for recipe, seeds in ARMS.items():
        for seed, ckpt_dir in seeds.items():
            rows = infer_arm(ckpt_dir)
            data[(recipe, seed)] = rows
            kd = _kernel_dev(ckpt_dir, "control" if recipe == "control" else "candidate", seed)
            la = summarize(rows)["accuracy"]
            if kd is not None:
                assert abs(kd - la) < 1e-6, f"{recipe} s{seed}: kernel dev {kd} != local {la} (checkpoint/metrics mismatch)"
            kernel_dev[(recipe, seed)] = kd

    report = {"deciding_seeds": list(DECIDING_SEEDS), "exploratory_seeds": list(EXPLORATORY),
              "per_seed": {}, "paired": {}, "per_type_deltas": {},
              "note_v2": "v2 kernel died of working-disk exhaustion mid probs_s44 (epoch 1); "
                         "s44 candidate + s45 rerun in v3 with per-arm checkpoint_latest cleanup; "
                         "control s43/s44 + candidate s43 recovered from v2 output"}
    for recipe, seeds in ARMS.items():
        for seed in seeds:
            s = summarize(data[(recipe, seed)])
            report["per_seed"][f"{recipe}_s{seed}"] = s
    pair_deltas = []
    for seed in DECIDING_SEEDS:
        a, b = data[("candidate", seed)], data[("control", seed)]
        sa, sb = summarize(a), summarize(b)
        boot = grouped_bootstrap_delta(a, b)
        pair_deltas.append(sa["accuracy"] - sb["accuracy"])
        report["paired"][f"s{seed}"] = {
            "candidate_acc": sa["accuracy"], "control_acc": sb["accuracy"],
            "delta_acc": sa["accuracy"] - sb["accuracy"],
            "delta_kl": sb["kl"] - sa["kl"], "delta_brier": sb["brier"] - sa["brier"],
            "delta_mae": (sb["mae"] - sa["mae"]) if sa["mae"] is not None else None,
            "bootstrap95_delta_acc": boot["ci95"],
            "by_type_delta": {t: sa["by_type"][t] - sb["by_type"][t] for t in sa["by_type"]},
        }
    deciding = [report["paired"][f"s{s}"] for s in DECIDING_SEEDS]
    report["summary_deciding"] = {
        "mean_delta_acc": sum(pair_deltas) / len(pair_deltas),
        "range_delta_acc": [min(pair_deltas), max(pair_deltas)],
        "wins": sum(1 for d in pair_deltas if d > 0),
        "mean_delta_kl": sum(d["delta_kl"] for d in deciding) / len(deciding),
    }
    if EXPLORATORY:
        report["exploratory"] = {f"candidate_s{s}": report["per_seed"][f"candidate_s{s}"]
                                 for s in EXPLORATORY}

    dest = ROOT / "runs/typed-decisions/e15_replication_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("| arm | acc | choice | noul | score | NLL | KL | Brier | MAE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, s in report["per_seed"].items():
        bt = s["by_type"]
        print(f"| {name} | {s['accuracy']:.4f} | {bt.get('choice', 0):.3f} | {bt.get('noul', 0):.3f} | "
              f"{bt.get('score', 0):.3f} | {s['nll']:.3f} | {s['kl']:.3f} | {s['brier']:.3f} | {s['mae']:.3f} |")
    sd = report["summary_deciding"]
    print(f"\nDECIDING pairs 42/43/44: mean d_acc {sd['mean_delta_acc']*100:+.2f} pp "
          f"(range {sd['range_delta_acc'][0]*100:+.2f}..{sd['range_delta_acc'][1]*100:+.2f}), "
          f"wins {sd['wins']}/3, mean ΔKL {sd['mean_delta_kl']:+.4f}")
    for seed in DECIDING_SEEDS:
        p = report["paired"][f"s{seed}"]
        ci = p["bootstrap95_delta_acc"]
        print(f"  s{seed}: d_acc {p['delta_acc']*100:+.2f} pp, grouped CI95 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]")
    print("saved ->", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
