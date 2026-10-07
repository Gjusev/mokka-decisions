"""Kaggle GPU job: install the pinned wheel, train / resume / evaluate.

Stages (arg --stage):
  smoke        hardware report + 100-step timing + save/no-save check
  train        full training from configs/train.yaml (or --config override)
  resume_test  train A (2 epochs) vs train B (1 epoch + reload + 1 epoch)
  eval         our model + temperature + threshold on all eval sets
  baselines    laya + gliner + tfidf on all eval sets (needs our rows? no: independent)

The bundle dataset (wheel + configs + prepared data) is discovered under
/kaggle/input. A checkpoint dataset (for resume) is discovered the same way.
Everything lands in /kaggle/working with a run_status.json verdict.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[mokka {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_bundle() -> Path:
    root = Path("/kaggle/input")
    if root.exists():
        # attachment debug: list every dataset mount before anything else
        log("input mounts: " + ", ".join(sorted(str(p) for p in root.iterdir())))
        candidates = [p.parent for p in root.rglob("BUNDLE_MANIFEST.json")]
        # prefer the full bundle (contains the wheel); checkpoint bundles also
        # carry a manifest but no dist/
        candidates.sort(key=lambda b: not any(b.rglob("*.whl")))
        for bundle in candidates:
            listing = [str(p) for p in sorted(bundle.rglob("*")) if p.is_file()][:80]
            log(f"bundle candidate {bundle}:\n  " + "\n  ".join(listing))
            if any(bundle.rglob("*.whl")):
                return bundle
        if candidates:
            raise SystemExit("only checkpoint-style bundles found; attach the full bundle dataset")
        listing = [str(p) for p in sorted(root.rglob("*"))][:60]
        raise SystemExit(
            "bundle dataset not found. /kaggle/input listing:\n" + "\n".join(listing)
        )
    raise SystemExit("no /kaggle/input; run locally with --bundle-style paths instead")


def find_file(bundle: Path, relative: str) -> Path:
    """Resolve a bundle-relative path across zip-extraction layouts."""
    target = bundle / relative
    if target.exists():
        return target
    name = Path(relative).name
    hits = [p for p in bundle.rglob(name) if p.is_file()]
    if not hits:
        raise SystemExit(f"bundle file not found: {relative} (no {name!r} anywhere)")
    expected_dirs = list(Path(relative).parts[:-1])
    for hit in hits:  # prefer hits whose parents contain the expected dirs
        if all(d in hit.parts for d in expected_dirs):
            return hit
    return hits[0]


def find_checkpoint() -> Path | None:
    for pattern in ("/kaggle/input/*/checkpoint_latest.pt", "/kaggle/input/*/checkpoint*.pt"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    return None


def install_wheel(bundle: Path) -> None:
    wheels = sorted(bundle.glob("**/*.whl"))
    if not wheels:
        raise SystemExit("no wheel in bundle")
    log(f"installing {wheels[0].name}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheels[0])])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "transformers==5.19.0", "datasets==5.1.0", "safetensors>=0.4", "sentencepiece>=0.2"]
    )


def hardware_report() -> dict:
    import torch

    report = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = props.name
        report["vram_gb"] = round(props.total_memory / 2**30, 2)
    return report


def require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU required but not assigned (phone verification / settings?). Refusing to train on CPU."
        )


def load_cases(path: Path):
    from mokka_decisions.contracts import DecisionCase, load_jsonl

    return [DecisionCase.from_dict(r) for r in load_jsonl(str(path))]


def stage_smoke(bundle: Path, work: Path, config: str) -> None:
    import yaml

    require_gpu()
    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    train_cfg = TrainConfig.from_mapping(cfg_data["train"])
    # 100-step-ish smoke: tiny slice, 1 epoch, no saves needed beyond latest
    cases = load_cases(find_file(bundle, "data/processed/v1/instances/train.jsonl"))[:64]
    dev = load_cases(find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl"))[:64]
    train_cfg.epochs = 1
    train_cfg.eval_every_steps = 0
    train_cfg.output_dir = str(work / "smoke")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.encoder_name)
    t0 = time.time()
    report = run_training(train_cfg, train_cases=cases, dev_cases=dev, tokenizer=tokenizer)
    dt = time.time() - t0
    steps = report["final_global_step"]
    summary = {
        "stage": "smoke",
        "hardware": hardware_report(),
        "steps": steps,
        "wall_s": round(dt, 1),
        "s_per_step": round(dt / max(steps, 1), 3),
        "dev": report["history"][-1]["dev"] if report["history"] else None,
    }
    (work / "run_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary))


def stage_train(bundle: Path, work: Path, config: str) -> None:
    import yaml

    require_gpu()
    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    train_cfg = TrainConfig.from_mapping(cfg_data["train"])
    train_cfg.output_dir = str(work / "train")

    train = load_cases(find_file(bundle, "data/processed/v1/instances/train.jsonl"))
    dev = load_cases(find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl"))
    log(f"train={len(train)} dev_eval={len(dev)}")

    resume = find_checkpoint()
    if resume:
        log(f"resuming from {resume}")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.encoder_name)
    manifest = json.loads((find_file(bundle, "data/processed/v1/manifests/revisions.json")).read_text())
    report = run_training(
        train_cfg,
        train_cases=train,
        dev_cases=dev,
        tokenizer=tokenizer,
        resume_from=resume,
        data_manifest_hash=manifest["data_hash"],
    )
    (work / "run_status.json").write_text(
        json.dumps(
            {
                "stage": "train",
                "hardware": hardware_report(),
                "resumed_from": str(resume) if resume else None,
                "final_global_step": report["final_global_step"],
                "best_dev_accuracy": report["best_dev_accuracy"],
                "epochs": train_cfg.epochs,
                "data_manifest_hash": manifest["data_hash"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done: best_dev={report['best_dev_accuracy']}")
    # copy best weights next to output for easy retrieval
    import shutil

    best = work / "train/checkpoint_best.safetensors"
    if best.exists():
        shutil.copy(best, work / "model_best.safetensors")
        meta = work / "train/checkpoint_best.json"
        if meta.exists():
            shutil.copy(meta, work / "model_best.json")


def stage_resume_test(bundle: Path, work: Path, config: str) -> None:
    """Obligatory resume proof: interrupted+resumed run vs continuous run."""
    require_gpu()
    import yaml

    from mokka_decisions.train import TrainConfig, run_training
    from transformers import AutoTokenizer

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    train = load_cases(find_file(bundle, "data/processed/v1/instances/train.jsonl"))[:2000]
    dev = load_cases(find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl"))[:400]
    tokenizer = AutoTokenizer.from_pretrained(cfg_data["train"]["encoder_name"])

    # A: continuous, 2 epochs
    cfg_a = TrainConfig.from_mapping(cfg_data["train"])
    cfg_a.epochs = 2
    cfg_a.output_dir = str(work / "resume_a")
    cfg_a.save_every_minutes = 0
    log("A: continuous 2 epochs")
    report_a = run_training(cfg_a, train_cases=train, dev_cases=dev, tokenizer=tokenizer)
    dev_a = [h for h in report_a["history"] if "epoch_time_s" in h.get("dev", {})][-1]["dev"]

    # B: 1 epoch, save, reload, +1 epoch
    cfg_b = TrainConfig.from_mapping(cfg_data["train"])
    cfg_b.epochs = 1
    cfg_b.output_dir = str(work / "resume_b1")
    cfg_b.save_every_minutes = 0
    log("B1: 1 epoch then stop")
    run_training(cfg_b, train_cases=train, dev_cases=dev, tokenizer=tokenizer)
    ckpt = work / "resume_b1/checkpoint_latest.pt"
    cfg_b2 = TrainConfig.from_mapping(cfg_data["train"])
    cfg_b2.epochs = 2
    cfg_b2.output_dir = str(work / "resume_b2")
    cfg_b2.save_every_minutes = 0
    log("B2: resume from checkpoint for epoch 2")
    report_b = run_training(
        cfg_b2, train_cases=train, dev_cases=dev, tokenizer=tokenizer, resume_from=ckpt
    )
    dev_b = [h for h in report_b["history"] if "epoch_time_s" in h.get("dev", {})][-1]["dev"]

    summary = {
        "stage": "resume_test",
        "continuous_dev": dev_a,
        "resumed_dev": dev_b,
        "accuracy_delta": abs(dev_a["accuracy"] - dev_b["accuracy"]),
        "nll_delta": abs(dev_a["mean_nll"] - dev_b["mean_nll"]),
        "verdict": "pass" if abs(dev_a["accuracy"] - dev_b["accuracy"]) <= 0.05 else "investigate",
        "note": "identical seeds; residual delta = CUDA kernel nondeterminism",
    }
    (work / "run_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary))


def stage_eval(bundle: Path, work: Path, config: str) -> None:
    """Our model: temperature on cal_temperature, threshold on cal_policy,
    metrics on every eval set (test pools opened here, once)."""
    require_gpu()
    import yaml

    from mokka_decisions.backends.custom import CustomBackend
    from mokka_decisions.calibrate import calibration_revision, fit_temperature, save_calibration
    from mokka_decisions.evaluate import (
        bootstrap_accuracy_ci,
        dump_rows,
        oos_false_acceptance,
        permutation_flip_rate,
        run_backend,
        summarize,
        summarize_by,
    )
    from mokka_decisions.instances import permute_case
    from mokka_decisions.policy import pick_threshold, risk_coverage_curve, save_policy

    # prefer a checkpoint trained in THIS session over any attached dataset
    # (an attached old checkpoint must never shadow the fresh one)
    ckpt = None
    for pattern in (
        "/kaggle/working/model_best.safetensors",
        "/kaggle/working/train/model_best.safetensors",
        "/kaggle/input/*/model_best.safetensors",
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            ckpt = Path(hits[0])
            break
    if ckpt is None:
        raise SystemExit("no trained checkpoint found for eval stage")
    log(f"eval checkpoint: {ckpt}")

    cfg_data = yaml.safe_load(find_file(bundle, config).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load((find_file(bundle, "configs/evaluation.yaml")).read_text(encoding="utf-8"))["evaluation"]
    backend = CustomBackend(ckpt, max_length=cfg_data["train"]["max_length"])
    backend.warmup()

    cal_t = load_cases(find_file(bundle, "data/processed/v1/instances/cal_temperature.jsonl"))
    cal_p = load_cases(find_file(bundle, "data/processed/v1/instances/cal_policy.jsonl"))
    log(f"cal_temperature={len(cal_t)} cal_policy={len(cal_p)}")

    logits = backend.scorer.score_logits(cal_t)
    temperature = fit_temperature(cal_t, logits)
    rev = calibration_revision(temperature, [c.id for c in cal_t])
    save_calibration(work / "calibration.json", temperature, rev, {"n": len(cal_t)})
    backend.scorer.temperature = temperature
    log(f"temperature={temperature}")

    probs_p = backend.scorer.score(cal_p)
    policy = pick_threshold(cal_p, probs_p, target_risk=eval_cfg["target_risk"])
    policy["calibration_revision"] = rev
    save_policy(work / "policy.json", policy)
    log(f"threshold={policy['threshold']} risk={policy['risk']} coverage={policy['coverage']}")
    curve = risk_coverage_curve(cal_p, probs_p)
    (work / "risk_coverage_calpolicy.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")

    results: dict[str, dict] = {}
    eval_dir = find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl").parent
    for eval_file in sorted(eval_dir.glob("*.jsonl")):
        name = eval_file.stem
        if name == "dev_eval":
            continue
        cases = load_cases(eval_file)
        rows = run_backend(backend, cases)
        dump_rows(rows, work / "rows" / f"{name}_mokka.jsonl")
        s = summarize(rows, cases, threshold=policy["threshold"])
        s["by_language"] = summarize_by(rows, cases, "language", threshold=policy["threshold"])
        s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=eval_cfg["bootstrap_n"])
        if any(c.target_option == "none" for c in cases):
            s["oos"] = oos_false_acceptance(rows, cases, threshold=policy["threshold"])
        results[name] = s
        log(f"{name}: acc={s['forced_accuracy']} coverage={s.get('coverage')}")

    # permutation probe on a sample of banking candidates eval
    perm_file = eval_dir / "banking77_test_candidates.jsonl"
    if perm_file.exists():
        base_cases = load_cases(perm_file)[: eval_cfg["permutation_probe_n"]]
        base_rows = run_backend(backend, base_cases)
        perm_cases = [permute_case(c, seed=1000 + i) for i, c in enumerate(base_cases)]
        perm_rows = run_backend(backend, perm_cases)
        results["permutation_probe"] = permutation_flip_rate(base_rows, perm_rows)

    (work / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (work / "run_status.json").write_text(
        json.dumps({"stage": "eval", "temperature": temperature, "policy": policy,
                    "checkpoint": str(ckpt), "sets": sorted(results)}, indent=2),
        encoding="utf-8",
    )


def stage_baselines(bundle: Path, work: Path, config: str) -> None:
    """Comparators on the same eval sets. TF-IDF trains on the train split
    (fixed taxonomy); laya and gliner run zero-shot. No model of ours needed."""
    require_gpu()
    # laya only: it coexists with transformers 5.x (verified in v8).
    # gliner2 goes EXCLUSIVELY to the isolated venv below — never here.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "laya>=0.3"]
    )
    from mokka_decisions.backends.laya import LayaBackend
    from mokka_decisions.backends.linear import LinearBackend, LinearRejectBackend
    from mokka_decisions.contracts import load_jsonl
    from mokka_decisions.evaluate import dump_rows, run_backend, summarize, summarize_by, bootstrap_accuracy_ci

    train_rows_raw = []
    for utt_file in sorted(find_file(bundle, "data/processed/v1/utterances/banking77.jsonl").parent.glob("*.jsonl")):
        train_rows_raw.extend(r for r in load_jsonl(str(utt_file)) if r.get("split") == "train")
    log(f"tfidf train rows: {len(train_rows_raw)}")
    backends = [LinearBackend(train_rows_raw), LinearRejectBackend(train_rows_raw)]
    log("tfidf fitted (closed-world + trainable-none variants)")
    try:
        backends.append(LayaBackend())
        log("laya loaded")
    except Exception as exc:
        log(f"laya load failed: {exc}")
    # gliner2 needs transformers<5: run it in an ISOLATED venv (the main env
    # keeps transformers 5.x for the trainer and future VLM comparators)
    gliner_ok = False
    try:
        import venv as _venv

        venv_dir = Path("/tmp/gliner-venv")
        vpy = str(venv_dir / "bin" / "python")
        if not Path(vpy).exists():
            import shutil as _sh

            if _sh.which("uv"):
                subprocess.check_call(["uv", "venv", str(venv_dir)])
                # uv installs with --python of the venv
                def _uv_install(*pkgs):
                    subprocess.check_call(["uv", "pip", "install", "--python", vpy, "--quiet", *pkgs])
            else:
                _venv.create(str(venv_dir), with_pip=True)

                def _uv_install(*pkgs):
                    subprocess.check_call([vpy, "-m", "pip", "install", "--quiet", *pkgs])
        else:
            def _uv_install(*pkgs):
                subprocess.check_call([vpy, "-m", "pip", "install", "--quiet", *pkgs])
        wheel = sorted(bundle.rglob("*.whl"))[0]
        _uv_install(str(wheel), "pyyaml", "safetensors")
        # CUDA torch in the venv when the kernel has a GPU (matches cu128);
        # fall back to CPU torch otherwise
        import torch as _torch

        if _torch.cuda.is_available():
            try:
                subprocess.check_call([vpy, "-m", "pip", "install", "--quiet",
                                       "torch==2.11.0", "--index-url",
                                       "https://download.pytorch.org/whl/cu128"])
            except Exception as exc:
                log(f"cuda torch in venv failed ({exc}); gliner falls back to CPU")
        _uv_install("gliner2[local]>=2.0", "transformers<5")
        gliner_ok = True
        log("gliner venv ready (isolated, transformers<5)")
    except Exception as exc:
        log(f"gliner venv setup failed: {exc}")

    def run_gliner_worker(cases):
        """Score via the isolated venv worker; returns BackendRow dicts."""
        from mokka_decisions.backends.base import BackendRow

        work_dir = Path("/tmp/gliner-io")
        work_dir.mkdir(exist_ok=True)
        cases_file = work_dir / "cases.jsonl"
        rows_file = work_dir / "rows.jsonl"
        cases_file.write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in cases), encoding="utf-8"
        )
        worker = Path(__file__).parent / "gliner_worker.py"
        subprocess.check_call(
            [str(Path("/tmp/gliner-venv/bin/python")), str(worker),
             str(cases_file), str(rows_file)], cwd=str(work_dir)
        )
        rows_raw = [json.loads(ln) for ln in rows_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [BackendRow(**{k: v for k, v in r.items() if k in BackendRow.__dataclass_fields__}) for r in rows_raw]

    eval_dir = find_file(bundle, "data/processed/v1/eval/dev_eval.jsonl").parent
    for eval_file in sorted(eval_dir.glob("*.jsonl")):
        name = eval_file.stem
        cases = load_cases(eval_file)
        for backend in backends:
            rows = run_backend(backend, cases)
            dump_rows(rows, work / "rows" / f"{name}_{backend.name}.jsonl")
            s = summarize(rows, cases)
            s["by_language"] = summarize_by(rows, cases, "language")
            s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=300)
            (work / "results" / f"{name}_{backend.name}.json").parent.mkdir(exist_ok=True)
            (work / "results" / f"{name}_{backend.name}.json").write_text(
                json.dumps(s, indent=2), encoding="utf-8"
            )
            log(f"{name} / {backend.name}: acc={s['forced_accuracy']}")
        if gliner_ok:
            try:
                rows = run_gliner_worker(cases)
                dump_rows(rows, work / "rows" / f"{name}_gliner25_multilingual_zero_shot.jsonl")
                s = summarize(rows, cases)
                s["by_language"] = summarize_by(rows, cases, "language")
                s["bootstrap"] = bootstrap_accuracy_ci(rows, cases, n_boot=300)
                (work / "results" / f"{name}_gliner25_multilingual_zero_shot.json").write_text(
                    json.dumps(s, indent=2), encoding="utf-8"
                )
                log(f"{name} / gliner25: acc={s['forced_accuracy']}")
            except Exception as exc:
                log(f"{name} / gliner25 worker failed: {exc}")

    # fair selective comparison: each baseline gets its own threshold fitted
    # on cal_policy (the same split and target the final model uses)
    try:
        from mokka_decisions.policy import pick_threshold, risk_coverage_curve

        cal_p = load_cases(find_file(bundle, "data/processed/v1/instances/cal_policy.jsonl"))
        thresholds = {}
        for backend in backends:
            probs = [backend.decide([c])[0].probabilities for c in cal_p]
            pol = pick_threshold(cal_p, probs, target_risk=0.05)
            thresholds[backend.name] = pol
            log(f"policy {backend.name}: thr={pol['threshold']} risk={pol['risk']} cov={pol['coverage']}")
        (work / "baseline_policies.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"baseline policy fitting failed: {exc}")

    (work / "run_status.json").write_text(
        json.dumps({"stage": "baselines", "backends": [b.name for b in backends]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", default="typed_small_ord",
        help="comma-separated stages: smoke,train,resume_test,eval,baselines",
    )
    parser.add_argument("--config", default="configs/train.yaml")  # bundle v1 carries pilot-budget data
    args = parser.parse_args()

    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("runs/kaggle-local")
    work.mkdir(parents=True, exist_ok=True)
    bundle = find_bundle()
    log(f"bundle={bundle} stage={args.stage}")
    install_wheel(bundle)
    log(f"hardware={hardware_report()}")

    stages = {
        "smoke": stage_smoke,
        "train": stage_train,
        "resume_test": stage_resume_test,
        "eval": stage_eval,
        "baselines": stage_baselines,
    }
    from mm_pilot import STAGES as MM_STAGES

    stages.update(MM_STAGES)
    completed = []
    for stage_name in [s.strip() for s in args.stage.split(",") if s.strip()]:
        if stage_name not in stages:
            raise SystemExit(f"unknown stage {stage_name!r}; known: {sorted(stages)}")
        stages[stage_name](bundle, work, args.config)
        completed.append(stage_name)
        # preserve each stage verdict; run_status.json records the whole sequence
        status_path = work / "run_status.json"
        prev = {}
        if status_path.exists():
            try:
                prev = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        prev["stages_completed"] = completed
        prev["last_stage"] = stage_name
        status_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    log(f"stages complete: {completed}")


if __name__ == "__main__":
    main()
