"""Typed-decisions adaptation arms (dispatched from the flat kernel).

Answers a specific causal question: how much of the zero-shot gap (36.15 %)
comes from training distribution vs the adapter? Arms, all on the SAME
typed-train data (data/typed-v1, split by original case):

  1. ``base_fresh``    mmBERT-base + our head, trained from upstream weights
  2. ``base_init``     same, initialised from our released routing checkpoint
  3. ``small_fresh``   mmBERT-small + our head — the equal-size, equal-data
                       comparison against Julia-1 (144.3M)

Each arm: train (dev selection inside typed dev), temperature on typed cal,
then the FROZEN protocol adapter on the pinned test parquet. Results carry
arm labels, hashes and effective settings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[typed {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_typed_root(bundle: Path) -> Path:
    for marker in bundle.rglob("typed-v1/manifest.json"):
        return marker.parent
    raise SystemExit("typed-v1 dataset not found in bundle (typed-v1/manifest.json)")


def _load(root: Path, split: str):
    from mokka_decisions.contracts import DecisionCase, load_jsonl

    rows = load_jsonl(str(root / "cases" / f"{split}.jsonl"))
    return [DecisionCase.from_dict(r) for r in rows]


def _train_arm(name: str, encoder: str, train, dev, work: Path, init_from: str = "", epochs: int = 3):
    import torch
    from transformers import AutoTokenizer

    from mokka_decisions.train import TrainConfig, run_training

    cfg = TrainConfig(
        encoder_name=encoder,
        max_length=512,
        microbatch=4,
        grad_accumulation=8,
        epochs=epochs,
        encoder_lr=2e-5,
        head_lr=1e-4,
        seed=42,
        amp=torch.cuda.is_available(),
        output_dir=str(work / name),
        init_from=init_from,
        max_options_per_step=32,
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    report = run_training(cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer)
    log(f"{name}: best_dev={report['best_dev_accuracy']}")
    return work / name / "checkpoint_best.safetensors", report


def stage_typed_arms(bundle: Path, work: Path, config: str) -> None:
    import glob

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_arms needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    cal = _load(root, "cal")
    log(f"typed data: train={len(train)} dev={len(dev)} cal={len(cal)}")

    # our released checkpoint as the init arm's starting point
    ckpts = sorted(glob.glob("/kaggle/input/**/model_best.safetensors", recursive=True))
    init_path = ckpts[0] if ckpts else ""
    log(f"init checkpoint: {init_path or '(none attached; base_init arm skipped)'}")

    arms = [("base_fresh", "jhu-clsp/mmBERT-base", "")]
    if init_path:
        arms.append(("base_init", "jhu-clsp/mmBERT-base", init_path))
    arms.append(("small_fresh", "jhu-clsp/mmBERT-small", ""))

    results = {}
    import glob as _g

    for name, encoder, init in arms:
        pretrained = sorted(_g.glob(f"/kaggle/input/*/arms/{name}/checkpoint_best.safetensors"))
        if pretrained:
            arm_dir = work / name
            arm_dir.mkdir(parents=True, exist_ok=True)
            import shutil as _sh

            _sh.copy2(pretrained[0], arm_dir / "checkpoint_best.safetensors")
            report = {"best_dev_accuracy": None, "final_step": None, "config_hash": ""}
            log(f"{name}: reusing attached pre-trained checkpoint {pretrained[0]}")
        else:
            best, report = _train_arm(name, encoder, train, dev, work, init_from=init)
        results[name] = {
            "encoder": encoder,
            "initialised_from": init or "upstream",
            "best_dev_accuracy": report["best_dev_accuracy"],
            "final_step": report["final_global_step"],
            "config_hash": report["config_hash"],
            "reused_checkpoint": bool(pretrained),
        }

    # ---- frozen-protocol evaluation on the pinned test parquet ----
    import math

    parquet = work / "typed-test-00000-of-00001.parquet"  # work dir: input is read-only
    if not parquet.exists():
        import urllib.request

        url = (
            "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/"
            "c76749ec58bd8c3d2ea706b31c333a9059c38f90/all/test-00000-of-00001.parquet"
        )
        parquet.write_bytes(urllib.request.urlopen(url, timeout=180).read())

    # inline frozen-protocol adapter (same construction as the local script)
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import OptionScorer, ScorerInference, load_checkpoint

    import pyarrow.parquet as pq

    cases_rows = pq.read_table(parquet).to_pylist()
    rows, metadata = [], []
    for case in cases_rows:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            rows.append((json.dumps(state, ensure_ascii=False), question["instructions"],
                         kind, [criteria[k] for k in keys]))
            metadata.append(dict(id=case["id"] + ":" + question_id, type=kind, keys=keys,
                                 gold=str(gold[question_id]["label"])))
    import collections

    counts = collections.Counter(r[2] for r in rows)
    assert len(cases_rows) == 400 and counts == {"choice": 600, "score": 800, "noul": 600}, counts

    eval_cases = [
        DecisionCase(
            id=m["id"], group_id=m["id"].split(":")[0], source="typed-decisions",
            language="en", domain="benchmark", state=sj, question=q,
            options=tuple(Option(id=f"o{i}", description=d) for i, d in enumerate(opts)),
            target_kind="single", target_option=f"o{m['keys'].index(m['gold'])}",
        )
        for (sj, q, k, opts), m in zip(rows, metadata)
    ]

    for name, encoder, _init in arms:
        from transformers import AutoTokenizer

        model = OptionScorer(encoder)
        load_checkpoint(model, work / name / "checkpoint_best.safetensors")
        tokenizer = AutoTokenizer.from_pretrained(encoder)
        scorer = ScorerInference(model, tokenizer, device="cuda", temperature=1.0,
                                 max_length=512, batch_decisions=16)
        probs_all = []
        t0 = time.time()
        for s in range(0, len(eval_cases), 16):
            probs_all.extend(scorer.score(eval_cases[s : s + 16]))
        stats = collections.defaultdict(lambda: dict(count=0, correct=0))
        mae = 0
        n_mae = 0
        nll = 0.0
        for case, m, probs in zip(eval_cases, metadata, probs_all):
            gold_idx = m["keys"].index(m["gold"])
            pred_idx = int(max(probs, key=probs.get)[1:])
            st = stats[m["type"]]
            st["count"] += 1
            st["correct"] += int(pred_idx == gold_idx)
            ordered = [probs[f"o{i}"] for i in range(len(m["keys"]))]
            nll += -math.log(max(ordered[gold_idx], 1e-12))
            if m["type"] == "score":
                mae += abs(sum(i * p for i, p in enumerate(ordered)) - gold_idx)
                n_mae += 1
        for st in stats.values():
            st["accuracy"] = st["correct"] / st["count"]
        total = sum(st["correct"] for st in stats.values())
        results[name]["test"] = {
            "overall": f"{total}/2000",
            "overall_accuracy": total / 2000,
            "by_type": dict(stats),
            "score_mae": mae / n_mae if n_mae else None,
            "mean_nll": nll / 2000,
            "eval_elapsed_s": round(time.time() - t0, 1),
            "protocol": "frozen v1.0 (json.dumps state, max_len 512, T=1.0, batch 16)",
        }
        log(f"{name} TEST: {total}/2000 = {total/2000:.4f} | {dict(stats)}")

    results["_meta"] = {
        "typed_train_manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        "comparison_reference": {
            "julia_reproduced_cpu": 0.7255,
            "mokka_base_zero_shot_asrun": 0.3615,
        },
        "arm_meaning": "equal data (typed-train) x encoder size x init",
    }
    (work / "typed_arms_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (work / "run_status.json").write_text(
        json.dumps({"stage": "typed_arms", "arms": [a[0] for a in arms]}, indent=2),
        encoding="utf-8",
    )


def stage_typed_a1(bundle: Path, work: Path, config: str) -> None:
    """A1 marker architecture on the same typed-train data as the A0 arms.

    One sequence per decision (state+question+marked options), scores from
    marker hidden states. Trained with the same recipe/splits; evaluated with
    the same frozen adapter construction. Reports sequence lengths, step time
    and permutation behaviour for the A0-vs-A1 cost/quality comparison.
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_a1 needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    log(f"typed A1 data: train={len(train)} dev={len(dev)}")

    from mokka_decisions.arch_a1 import MarkerScorer, train_a1, collate_a1, forward_a1

    model = MarkerScorer("jhu-clsp/mmBERT-base")
    t0 = _time.time()
    report = train_a1(
        model, train, dev,
        output_dir=str(work / "a1_base"), epochs=3, microbatch=4, grad_accumulation=8,
        device="cuda",
    )
    log(f"a1 trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")

    # reload best and run the frozen-protocol eval (same construction as arms)
    from mokka_decisions.contracts import DecisionCase, Option
    from mokka_decisions.model import load_checkpoint
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(work / "a1_base" / "checkpoint_best.safetensors")))
    model = model.cuda().eval()

    import pyarrow.parquet as pq
    import urllib.request

    parquet = work / "typed-test-00000-of-00001.parquet"  # work dir: input is read-only
    if not parquet.exists():
        url = (
            "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/"
            "c76749ec58bd8c3d2ea706b31c333a9059c38f90/all/test-00000-of-00001.parquet"
        )
        parquet.write_bytes(urllib.request.urlopen(url, timeout=180).read())
    import collections
    import math

    cases_rows = pq.read_table(parquet).to_pylist()
    eval_cases, metadata = [], []
    for case in cases_rows:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        for question_id, question in json.loads(case["questions"]).items():
            kind = question["type"]
            criteria = question.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if isinstance(criteria, list):
                criteria = {str(i): v for i, v in enumerate(criteria)}
            keys = ["false", "true"] if kind == "noul" else list(criteria)
            eval_cases.append(DecisionCase(
                id=f"{case['id']}:{question_id}", group_id=case["id"], source="typed-decisions",
                language="en", domain="benchmark",
                state=json.dumps(state, ensure_ascii=False), question=question["instructions"],
                options=tuple(Option(id=f"o{i}", description=criteria[k]) for i, k in enumerate(keys)),
                target_kind="single", target_option=f"o{keys.index(str(gold[question_id]['label']))}",
            ))
            metadata.append(dict(id=eval_cases[-1].id, type=kind, keys=keys,
                                 gold=str(gold[question_id]["label"])))

    stats = collections.defaultdict(lambda: dict(count=0, correct=0))
    mae = n_mae = 0
    nll = 0.0
    t0 = _time.time()
    skipped_truncation = 0
    latencies = []
    for s in range(0, len(eval_cases), 8):
        chunk = eval_cases[s : s + 8]
        t1 = _time.perf_counter()
        batch = collate_a1(model, chunk, "cuda")
        logits = forward_a1(model, batch).float()
        torch.cuda.synchronize()
        if s < 100:
            latencies.append((_time.perf_counter() - t1) * 1000 / len(chunk))
        preds = logits.argmax(-1).tolist()
        for i, (case, m) in enumerate(zip(chunk, metadata[s : s + 8])):
            if batch["targets"][i].item() == -100:
                skipped_truncation += 1
                stats[m["type"]]["count"] += 1
                continue  # truncated markers: counted as failures (unusable)
            gold_idx = m["keys"].index(m["gold"])
            correct = preds[i] == gold_idx
            st = stats[m["type"]]
            st["count"] += 1
            st["correct"] += int(correct)
    for st in stats.values():
        st["accuracy"] = st["correct"] / st["count"] if st["count"] else None
    total_correct = sum(st["correct"] for st in stats.values())
    result = {
        "arm": "a1_base",
        "label": "adapted (equal data, marker architecture)",
        "best_dev_accuracy": report["best_dev_accuracy"],
        "test": {
            "overall": f"{total_correct}/2000",
            "overall_accuracy": total_correct / 2000,
            "by_type": dict(stats),
            "skipped_by_truncation": skipped_truncation,
            "eval_elapsed_s": round(_time.time() - t0, 1),
            "latency_ms_batch1_p50": sorted(latencies)[len(latencies) // 2] if latencies else None,
            "protocol": "frozen v1.0 adapter construction",
        },
        "history": report["history"],
    }
    (work / "typed_a1_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"A1 TEST: {total_correct}/2000 = {total_correct/2000:.4f} skipped={skipped_truncation}")
    (work / "run_status.json").write_text(
        json.dumps({"stage": "typed_a1", "overall": total_correct / 2000}, indent=2), encoding="utf-8"
    )


def stage_typed_ord(bundle: Path, work: Path, config: str, epochs: int = 3, out: str = "base_ord") -> None:
    """H1 experiment: A0 base + ordinal soft-target loss on score rows.

    Hypothesis (from dev error diagnosis): score-type rows fail by confident
    adjacent-level confusions; a unimodal soft target (tau=1) on score rows
    should improve score accuracy and MAE without touching anything else.
    Control: base_fresh (same data, init, recipe, seeds) -> dev comparison.
    typed_ord2 = H2: same recipe, epochs extended only (dev curve not saturated).
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_ord needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    log(f"typed_ord data: train={len(train)} dev={len(dev)}")

    # question types come from the rebuilt typed-v1 rows
    types_path = root / "cases" / "train.jsonl"
    ordinal_ids = set()
    import json as _json

    for line in types_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = _json.loads(line)
            if row.get("question_type") == "score":
                ordinal_ids.add(row["id"])
    log(f"ordinal (score) train rows: {len(ordinal_ids)}")

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg = TrainConfig(
        encoder_name="jhu-clsp/mmBERT-base",
        max_length=512, microbatch=4, grad_accumulation=8, epochs=epochs,
        encoder_lr=2e-5, head_lr=1e-4, seed=42, amp=True,
        output_dir=str(work / out), max_options_per_step=32,
    )
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    t0 = _time.time()
    report = run_training(
        cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
        ordinal_case_ids=ordinal_ids, ordinal_tau=1.0,
    )
    log(f"base_ord trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")

    # dev diagnosis by question type (frozen construction)
    import collections
    import math

    dev_rows = [ _json.loads(l) for l in (root / "cases" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() ]
    types = {r["id"]: r.get("question_type", "?") for r in dev_rows}
    from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference
    from safetensors.torch import load_file

    m = OptionScorer("jhu-clsp/mmBERT-base")
    load_checkpoint(m, work / out / "checkpoint_best.safetensors")
    sc = ScorerInference(m, tokenizer, device="cuda", temperature=1.0, max_length=512, batch_decisions=16)
    typed_dev = [c for c in dev if c.target_option]
    probs = sc.score(typed_dev)
    stats = collections.defaultdict(lambda: dict(n=0, ok=0))
    mae = n_mae = 0
    for c, p_ in zip(typed_dev, probs):
        pred = max(p_, key=p_.get)
        qt = types.get(c.id, "?")
        stats[qt]["n"] += 1
        stats[qt]["ok"] += int(pred == c.target_option)
        if qt == "score":
            idx = c.option_ids.index(c.target_option)
            ev = sum(i * p_[oid] for i, oid in enumerate(c.option_ids))
            mae += abs(ev - idx)
            n_mae += 1
    result = {
        "arm": out,
        "hypothesis": "unimodal soft targets on score rows fix confident adjacent-level confusions",
        "control": "base_fresh (same data/init/recipe)",
        "epochs": epochs,
        "ordinal_train_rows": len(ordinal_ids),
        "best_dev_accuracy": report["best_dev_accuracy"],
        "dev_by_type": {k: {"acc": v["ok"] / v["n"], "n": v["n"]} for k, v in stats.items()},
        "dev_score_mae": mae / n_mae if n_mae else None,
        "history": report["history"],
    }
    (work / f"{out}_results.json").write_text(_json.dumps(result, indent=2), encoding="utf-8")
    log(f"{out} dev by type: {result['dev_by_type']} mae={result['dev_score_mae']}")
    (work / "run_status.json").write_text(
        _json.dumps({"stage": f"typed_ord:{out}", "best_dev": report["best_dev_accuracy"]}, indent=2),
        encoding="utf-8",
    )


def stage_typed_small_ord(bundle: Path, work: Path, config: str, epochs: int = 4, out: str = "small_ord") -> None:
    """Recipe transfer: the dev-winning ordinal loss on mmBERT-small.

    Equal-parameter comparison vs Julia-1 (140M vs 144M) with the recipe that
    won on Base. Same data, seed and protocol as small_fresh (control).
    """
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_small_ord needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    import json as _json

    ordinal_ids = {
        _json.loads(line)["id"]
        for line in (root / "cases" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and _json.loads(line).get("question_type") == "score"
    }
    log(f"small_ord: train={len(train)} dev={len(dev)} ordinal={len(ordinal_ids)}")

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg = TrainConfig(
        encoder_name="jhu-clsp/mmBERT-small",
        max_length=512, microbatch=4, grad_accumulation=8, epochs=epochs,
        encoder_lr=2e-5, head_lr=1e-4, seed=42, amp=True,
        output_dir=str(work / out), max_options_per_step=32,
    )
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    t0 = _time.time()
    report = run_training(
        cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
        ordinal_case_ids=ordinal_ids, ordinal_tau=1.0,
    )
    log(f"small_ord trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")

    # dev by type
    import collections

    dev_rows = [_json.loads(l) for l in (root / "cases" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    types = {r["id"]: r.get("question_type", "?") for r in dev_rows}
    from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference
    from safetensors.torch import load_file

    m = OptionScorer("jhu-clsp/mmBERT-small")
    load_checkpoint(m, work / out / "checkpoint_best.safetensors")
    sc = ScorerInference(m, tokenizer, device="cuda", temperature=1.0, max_length=512, batch_decisions=16)
    typed_dev = [c for c in dev if c.target_option]
    probs = sc.score(typed_dev)
    stats = collections.defaultdict(lambda: dict(n=0, ok=0))
    mae = n_mae = 0
    for c, p_ in zip(typed_dev, probs):
        pred = max(p_, key=p_.get)
        qt = types.get(c.id, "?")
        stats[qt]["n"] += 1
        stats[qt]["ok"] += int(pred == c.target_option)
        if qt == "score":
            idx = c.option_ids.index(c.target_option)
            ev = sum(i * p_[oid] for i, oid in enumerate(c.option_ids))
            mae += abs(ev - idx)
            n_mae += 1
    result = {
        "arm": out,
        "hypothesis": "ordinal-loss recipe transfers from Base to Small",
        "control": "small_fresh (dev 0.6386, test 0.5345)",
        "epochs": epochs,
        "best_dev_accuracy": report["best_dev_accuracy"],
        "dev_by_type": {k: {"acc": v["ok"] / v["n"], "n": v["n"]} for k, v in stats.items()},
        "dev_score_mae": mae / n_mae if n_mae else None,
        "history": report["history"],
    }
    (work / f"{out}_results.json").write_text(_json.dumps(result, indent=2), encoding="utf-8")
    log(f"{out} dev by type: {result['dev_by_type']} mae={result['dev_score_mae']}")
    (work / "run_status.json").write_text(
        _json.dumps({"stage": f"typed_small_ord:{out}", "best_dev": report["best_dev_accuracy"]}, indent=2),
        encoding="utf-8",
    )


def stage_typed_e15(bundle: Path, work: Path, config: str, seed: int = 42, arms: str = "both") -> None:
    """E15: dataset-native score supervision vs the artificial ordinal target.

    seed/arms parametrise the matched-seed replication (seed 42 ran as v1);
    arms: "both" | "control" | "probs".

    Control (e15_control): the base_ord recipe exactly — artificial unimodal
    targets exp(-|j-gold|/tau=1) on score rows, hard CE on choice/noul.
    Candidate (e15_probs): identical data, init, batch, optimizer, steps,
    LR schedule and selection; ONLY the score-row target vector changes to
    the original gold.probabilities of the train split (soft CE).
    Pre-fixed selection: best dev accuracy (same criterion as every arm).
    Recorded metrics per arm: accuracy by type, mean NLL (hard label),
    KL(gold||model) on all dev rows, multiclass Brier, score MAE.
    Dev probabilities are used for METRICS only, never for training.
    """
    import collections
    import math
    import time as _time

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("typed_e15 needs the GPU that was not assigned")
    root = _find_typed_root(bundle)
    train = _load(root, "train")
    dev = _load(root, "dev")
    import json as _json

    def _load_probs(split: str) -> dict:
        rows = [
            _json.loads(l)
            for l in (root / "cases" / f"probabilities_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        return {r["id"]: r for r in rows}

    probs_train = _load_probs("train")
    probs_dev = _load_probs("dev")
    qtypes = {
        _json.loads(l)["id"]: _json.loads(l).get("question_type", "?")
        for l in (root / "cases" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    # candidate targets: dataset-native distributions on score rows only
    soft_targets = {}
    for c in train:
        if qtypes.get(c.id) == "score":
            row = probs_train.get(c.id)
            if row is None or len(row["probabilities"]) != len(c.options):
                raise SystemExit(f"probabilities misaligned for {c.id}")
            s = sum(row["probabilities"])
            if abs(s - 1.0) > 1e-3:
                raise SystemExit(f"probabilities not normalised for {c.id} (sum={s})")
            soft_targets[c.id] = row["probabilities"]
    ordinal_ids = {i for i, t in qtypes.items() if t == "score"}
    log(f"typed_e15: train={len(train)} dev={len(dev)} score_rows={len(ordinal_ids)} "
        f"soft_targets={len(soft_targets)} dev_probs={len(probs_dev)}")
    assert set(soft_targets) == ordinal_ids, "every score train row must carry its distribution"

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")

    def _arm(name: str, *, ordinal: bool) -> dict:
        cfg = TrainConfig(
            encoder_name="jhu-clsp/mmBERT-base",
            max_length=512, microbatch=4, grad_accumulation=8, epochs=3,
            encoder_lr=2e-5, head_lr=1e-4, seed=seed, amp=True,
            output_dir=str(work / name), max_options_per_step=32,
        )
        t0 = _time.time()
        report = run_training(
            cfg, train_cases=train, dev_cases=dev, tokenizer=tokenizer,
            ordinal_case_ids=ordinal_ids if ordinal else None, ordinal_tau=1.0,
            soft_targets=None if ordinal else soft_targets,
        )
        log(f"{name} trained in {_time.time()-t0:.0f}s best_dev={report['best_dev_accuracy']}")
        return report

    name_c = f"e15_control_s{seed}"
    name_p = f"e15_probs_s{seed}"
    report_c = _arm(name_c, ordinal=True) if arms in ("both", "control") else None
    report_p = _arm(name_p, ordinal=False) if arms in ("both", "probs") else None

    # ---- dev metrics for both arms (checkpoint_best of each) ----
    from mokka_decisions.model import OptionScorer, ScorerInference, load_checkpoint

    def _metrics(name: str) -> dict:
        m = OptionScorer("jhu-clsp/mmBERT-base")
        load_checkpoint(m, work / name / "checkpoint_best.safetensors")
        sc = ScorerInference(m, tokenizer, device="cuda", temperature=1.0, max_length=512, batch_decisions=16)
        typed_dev = [c for c in dev if c.target_option]
        probs = sc.score(typed_dev)
        stats = collections.defaultdict(lambda: dict(n=0, ok=0))
        nll = n = 0
        kl = n_kl = 0
        brier = n_br = 0
        mae = n_mae = 0
        for c, p_ in zip(typed_dev, probs):
            qt = probs_dev[c.id]["question_type"] if c.id in probs_dev else qtypes.get(c.id, "?")
            vec = probs_dev[c.id]["probabilities"] if c.id in probs_dev else None
            gold_idx = c.option_ids.index(c.target_option)
            pv = [p_[oid] for oid in c.option_ids]
            stats[qt]["n"] += 1
            stats[qt]["ok"] += int(max(p_, key=p_.get) == c.target_option)
            eps = 1e-12
            nll += -math.log(max(pv[gold_idx], eps)); n += 1
            if vec is not None and len(vec) == len(pv):
                kl += sum(v * (math.log(max(v, eps)) - math.log(max(pi, eps)))
                          for v, pi in zip(vec, pv) if v > 0)
                n_kl += 1
                brier += sum((pi - (1.0 if i == gold_idx else 0.0)) ** 2 for i, pi in enumerate(pv))
                n_br += 1
            if qt == "score":
                ev = sum(i * pi for i, pi in enumerate(pv))
                mae += abs(ev - gold_idx); n_mae += 1
        return {
            "dev_by_type": {k: {"acc": v["ok"] / v["n"], "n": v["n"]} for k, v in stats.items()},
            "mean_nll": nll / n if n else None,
            "mean_kl_gold_vs_model": kl / n_kl if n_kl else None,
            "mean_brier_multiclass": brier / n_br if n_br else None,
            "dev_score_mae": mae / n_mae if n_mae else None,
        }

    result = {
        "experiment": "E15 dataset-native score supervision",
        "seed": seed,
        "selection_criterion": "best dev accuracy (pre-fixed)",
        "control": ({"arm": name_c, "best_dev_accuracy": report_c["best_dev_accuracy"],
                     "history": report_c["history"], **_metrics(name_c)}
                    if report_c else None),
        "candidate": ({"arm": name_p, "best_dev_accuracy": report_p["best_dev_accuracy"],
                       "history": report_p["history"], **_metrics(name_p)}
                      if report_p else None),
        "note": "candidate differs ONLY in score-row targets (original probabilities via soft CE); "
                "choice/noul untouched; same data/init/batch/optimizer/steps/schedule/seed.",
    }
    (work / f"e15_s{seed}_results.json").write_text(_json.dumps(result, indent=2), encoding="utf-8")
    log(f"e15 seed={seed}: " + " ".join(
        f"{a['arm']} dev={a['best_dev_accuracy']:.4f}" for a in (result["control"], result["candidate"]) if a))
    (work / "run_status.json").write_text(
        _json.dumps({"stage": f"typed_e15:seed{seed}", "seed": seed,
                     "control_dev": result["control"]["best_dev_accuracy"] if result["control"] else None,
                     "candidate_dev": result["candidate"]["best_dev_accuracy"] if result["candidate"] else None},
                    indent=2),
        encoding="utf-8",
    )


def stage_typed_e15_seeds(bundle: Path, work: Path, config: str) -> None:
    """Matched-seed replication of E15 (commission 2026-10-08, step 3).

    Seed 42 already ran in v1 (collected). Here: control seeds 43,44 and
    candidate seeds 43,44,45 -> 3 seeds per recipe, matched pairs at
    42/43/44 plus one extra candidate. Same kernel session so the paired
    comparison avoids cross-session GPU variance.
    """
    for seed in (43, 44):
        stage_typed_e15(bundle, work, config, seed=seed)
    stage_typed_e15(bundle, work, config, seed=45, arms="probs")


STAGES = {"typed_arms": stage_typed_arms, "typed_a1": stage_typed_a1,
          "typed_ord": stage_typed_ord, "typed_small_ord": stage_typed_small_ord,
          "typed_e15": stage_typed_e15, "typed_e15_seeds": stage_typed_e15_seeds,
          # H2 (controlled, one variable): the dev curves were not saturated at
          # 4/8 epochs; extend epochs only — data/init/loss/seeds unchanged.
          "typed_ord2": lambda b, w, c: stage_typed_ord(b, w, c, epochs=6, out="base_ord6"),
          "typed_small_ord2": lambda b, w, c: stage_typed_small_ord(b, w, c, epochs=8, out="small_ord8")}
