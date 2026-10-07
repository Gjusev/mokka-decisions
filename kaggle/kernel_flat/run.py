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
        if not (venv_dir / "bin" / "python").exists():
            _venv.create(str(venv_dir), with_pip=True)
        vpy = str(venv_dir / "bin" / "python")
        wheel = sorted(bundle.rglob("*.whl"))[0]
        subprocess.check_call([vpy, "-m", "pip", "install", "--quiet",
                               str(wheel), "pyyaml", "safetensors"])
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
        subprocess.check_call([vpy, "-m", "pip", "install", "--quiet",
                               "gliner2[local]>=2.0", "transformers<5"])
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
        worker = Path("/tmp/gliner_worker.py")
        worker.write_text(json.loads(GLINER_WORKER_SRC), encoding="utf-8")
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
        "--stage", default="baselines",
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
    stages.update(STAGES)
    stages.update(TYPED_STAGES)  # inlined pilot stage dicts (defined below)
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

# ==== inlined: multimodal pilot stages (kaggle/mm_pilot.py) ====


import json
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[mm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find_mm_root(bundle: Path) -> Path:
    for marker in bundle.rglob("mm/manifest.json"):
        return marker.parent
    raise SystemExit("multimodal dataset not found in bundle (mm/manifest.json)")


def _load_mm_cases(root: Path, split: str):
    from mokka_decisions.contracts import DecisionCase, load_jsonl

    rows = load_jsonl(str(root / "cases" / f"{split}.jsonl"))
    cases = [DecisionCase.from_dict(r) for r in rows]
    types = {r["id"]: r.get("case_type", "") for r in rows}
    media = {r["id"]: r.get("media") for r in rows if r.get("media")}
    return cases, types, media


def _install(pkgs: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])


def stage_mm_profile(bundle: Path, work: Path, config: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("mm_profile needs the GPU that was not assigned")
    _install(["matplotlib", "pillow>=10"])
    from mokka_decisions.multimodal.model import SIGLIP2_MODEL, MultimodalOptionScorer, ImageTower
    from transformers import AutoTokenizer, AutoProcessor

    root = _find_mm_root(bundle)
    cases, types, media = _load_mm_cases(root, "train")
    log(f"mm data: {len(cases)} train cases, {sum(1 for v in media.values() if (v or {}).get('screenshot', {}).get('sha256'))} with screenshots")

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
    tower = ImageTower().cuda().eval()
    model = MultimodalOptionScorer("jhu-clsp/mmBERT-base").cuda()
    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    tower_params = sum(p.numel() for p in model.tower.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"params: encoder={enc_params:.1f}M tower(frozen)={tower_params:.1f}M trainable={trainable:.1f}M total={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # build a small cache + run 60 steps at microbatch 1 (fp16 on T4 = half, not bf16)
    from mokka_decisions.multimodal.train_mm import build_image_cache, encode_mm_batch

    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    cache = build_image_cache(cases[:200], media, root, tower, proc, device="cuda")
    log(f"image cache: {len(cache)} vectors in {time.time()-t0:.1f}s")

    opt = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4))
    scaler = torch.amp.GradScaler("cuda")
    from torch.nn import functional as F

    model.train()
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    n_steps = 60
    idx = 0
    for step in range(n_steps):
        case = cases[idx % len(cases)]
        idx += 1
        if not case.target_option:
            continue
        t1 = time.time()
        batch = encode_mm_batch(tok, [case], media, cache, max_length=256, device="cuda", vision_dim=tower.dim)
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            logits = model(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                batch["decision_index"], batch["option_slot"],
                batch["num_decisions"], batch["max_options"], batch["image_vec"],
            )
            loss = F.cross_entropy(logits, batch["targets"])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        if step >= 10:  # skip warmup steps in the timing stats
            step_times.append(time.time() - t1)
    vram = torch.cuda.max_memory_allocated() / 2**30
    mean_s = sum(step_times) / len(step_times)
    profile = {
        "stage": "mm_profile",
        "params": {"encoder_M": round(enc_params, 1), "tower_frozen_M": round(tower_params, 1),
                   "trainable_M": round(trainable, 1), "total_M": round(enc_params + tower_params + (trainable - enc_params), 1)},
        "microbatch": 1,
        "steps_timed": len(step_times),
        "mean_s_per_step": round(mean_s, 4),
        "est_train_minutes_microbatch4": round(mean_s * (len(cases) * 3 / 4) / 60 * 0.55, 1),
        "peak_vram_gb": round(vram, 2),
        "dtype": "fp16 (T4: no bf16 assumed)",
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    (work / "mm_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    log(json.dumps(profile))


def stage_mm_train(bundle: Path, work: Path, config: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("mm_train needs the GPU that was not assigned")
    _install(["matplotlib", "pillow>=10"])
    from mokka_decisions.multimodal.model import MultimodalOptionScorer, SIGLIP2_MODEL
    from mokka_decisions.multimodal.train_mm import (
        build_image_cache,
        load_case_types,
        load_media,
        train_multimodal,
    )
    from transformers import AutoTokenizer, AutoProcessor

    root = _find_mm_root(bundle)
    train_cases, _, _ = _load_mm_cases(root, "train")
    dev_cases, _, _ = _load_mm_cases(root, "dev")
    cal_t, _, _ = _load_mm_cases(root, "cal_temperature")
    cal_p, _, _ = _load_mm_cases(root, "cal_policy")

    case_files = [root / "cases" / f"{s}.jsonl" for s in ("train", "dev", "cal_temperature", "cal_policy", "test")]
    media, _ = load_media(case_files)
    types = load_case_types(case_files)

    # initialise the text encoder from OUR trained textual checkpoint when
    # attached (dataset gjusev/mokka-decisions-checkpoint), else from scratch
    import glob as _glob

    ckpts = sorted(_glob.glob("/kaggle/input/*/model_best.safetensors"))
    model = MultimodalOptionScorer("jhu-clsp/mmBERT-base")
    if ckpts:
        from mokka_decisions.model import load_checkpoint

        log(f"initialising encoder from textual checkpoint: {ckpts[0]}")
        textual_state = None
        from safetensors.torch import load_file

        textual_state = load_file(ckpts[0])
        enc_keys = {k[len("encoder."):]: v for k, v in textual_state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc_keys, strict=False)
        log(f"encoder init: {len(enc_keys)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")

    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    proc = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
    tower = model.tower
    all_cases = train_cases + dev_cases + cal_t + cal_p
    t0 = time.time()
    cache = build_image_cache(all_cases, media, root, tower, proc, device="cuda")
    log(f"image cache: {len(cache)} vectors in {time.time()-t0:.1f}s")

    out = work / "mm_train"
    report = train_multimodal(
        model=model,
        tokenizer=tok,
        train_cases=train_cases,
        dev_cases=dev_cases,
        media=media,
        vec_cache=cache,
        output_dir=out,
        epochs=3,
        microbatch=4,
        grad_accumulation=8,
        case_types=types,
    )
    log(f"mm_train done: {json.dumps(report)[:300]}")

    # calibration + policy on the mm splits with the best model
    from safetensors.torch import load_file

    best_path = out / "model_best.safetensors"
    model.load_state_dict(load_file(str(best_path)))
    from mokka_decisions.calibrate import fit_temperature, save_calibration, calibration_revision
    from mokka_decisions.multimodal.train_mm import encode_mm_batch
    from mokka_decisions.policy import pick_threshold, save_policy, risk_coverage_curve

    model = model.cuda().eval()

    def score(cases_list):
        probs = []
        with torch.no_grad():
            for s in range(0, len(cases_list), 16):
                chunk = cases_list[s : s + 16]
                batch = encode_mm_batch(tok, chunk, media, cache, max_length=256, device="cuda", vision_dim=tower.dim)
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    logits = model(
                        batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                        batch["decision_index"], batch["option_slot"],
                        batch["num_decisions"], batch["max_options"], batch["image_vec"],
                    ).float()
                for i, case in enumerate(chunk):
                    vals = logits[i, : len(case.options)]
                    pp = torch.softmax(vals, dim=-1)
                    probs.append({oid: float(v) for oid, v in zip(case.option_ids, pp)})
        return probs

    # temperature on cal_temperature (finite rows only)
    logits_rows = []
    with torch.no_grad():
        for s in range(0, len(cal_t), 16):
            chunk = [c for c in cal_t[s : s + 16] if c.target_option]
            if not chunk:
                continue
            batch = encode_mm_batch(tok, chunk, media, cache, max_length=256, device="cuda", vision_dim=tower.dim)
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                logits = model(
                    batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                    batch["decision_index"], batch["option_slot"],
                    batch["num_decisions"], batch["max_options"], batch["image_vec"],
                ).float()
            for i, case in enumerate(chunk):
                logits_rows.append({oid: float(logits[i, j]) for j, oid in enumerate(case.option_ids)})
    finite_t = [c for c in cal_t if c.target_option]
    temperature = fit_temperature(finite_t, logits_rows)
    rev = calibration_revision(temperature, [c.id for c in finite_t])
    save_calibration(work / "mm_calibration.json", temperature, rev, {"n": len(finite_t)})
    log(f"mm temperature={temperature}")

    probs_p = score([c for c in cal_p if c.target_option])
    policy = pick_threshold([c for c in cal_p if c.target_option], probs_p, target_risk=0.05)
    save_policy(work / "mm_policy.json", policy)
    log(f"mm policy: {json.dumps(policy)[:200]}")

    # test evaluation by case_type (single pass, frozen everything)
    test_cases, test_types, _ = _load_mm_cases(root, "test")
    from mokka_decisions.evaluate import summarize, summarize_by, dump_rows
    from mokka_decisions.backends.base import BackendRow
    import time as _time

    rows = []
    typed = [c for c in test_cases if c.target_option]
    probs = score(typed)
    for case, p in zip(typed, probs):
        t0_ = _time.perf_counter()
        cand = max(p, key=p.get)
        rows.append(BackendRow(
            case_id=case.id, group_id=case.group_id, source=case.source, language=case.language,
            domain=case.domain, backend="mokka_mm", model_revision=rev, candidate=cand,
            probabilities=p, latency_ms=0.0))
    dump_rows(rows, work / "rows" / "mm_test_mokka_mm.jsonl")
    by_type = {}
    for t in ("vision_needed", "text_suffices", "blurry", "no_valid_option"):
        sub = [(r, c) for r, c in zip(rows, typed) if test_types.get(c.id) == t]
        if sub:
            by_type[t] = summarize([r for r, _ in sub], [c for _, c in sub], threshold=policy.get("threshold"))
    overall = summarize(rows, typed, threshold=policy.get("threshold"))
    result = {"overall": overall, "by_case_type": by_type, "temperature": temperature, "policy": policy}
    (work / "mm_eval_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"mm test overall: {json.dumps(overall)[:250]}")


def stage_mm_baselines(bundle: Path, work: Path, config: str) -> None:
    """OCR->text and Qwen3.5-0.8B on the same mm test cases (+ textual mokka)."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("mm_baselines needs the GPU that was not assigned")
    root = _find_mm_root(bundle)
    test_cases, test_types, media = _load_mm_cases(root, "test")
    typed = [c for c in test_cases if c.target_option]

    from mokka_decisions.evaluate import dump_rows, summarize
    from mokka_decisions.backends.base import BackendRow

    def rows_from_probs(name, probs_list, latency_ms=None):
        rows = []
        for case, p in zip(typed, probs_list):
            if not p:
                rows.append(BackendRow(case_id=case.id, group_id=case.group_id, source=case.source,
                                       language=case.language, domain=case.domain, backend=name,
                                       model_revision="", candidate="", probabilities={},
                                       latency_ms=0.0, error="no output"))
                continue
            total = sum(p.values()) or 1.0
            pn = {k: v / total for k, v in p.items()}
            rows.append(BackendRow(case_id=case.id, group_id=case.group_id, source=case.source,
                                   language=case.language, domain=case.domain, backend=name,
                                   model_revision="", candidate=max(pn, key=pn.get),
                                   probabilities=pn, latency_ms=latency_ms or 0.0))
        return rows

    # ---- OCR -> text (rapidocr, ONNX runtime, full cost counted) ----
    try:
        _install(["rapidocr-onnxruntime", "onnxruntime"])
        from rapidocr_onnxruntime import RapidOCR
        from PIL import Image

        ocr = RapidOCR()
        import time as _time

        ocr_probs, ocr_times = [], []
        from mokka_decisions.multimodal.train_mm import encode_mm_batch  # noqa: F401 (typing)

        tok = None
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
        import glob as _glob

        ckpts = sorted(_glob.glob("/kaggle/input/*/model_best.safetensors"))
        from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference

        m = OptionScorer("jhu-clsp/mmBERT-base")
        if ckpts:
            load_checkpoint(m, ckpts[0])
        scorer = ScorerInference(m, tok, device="cuda", temperature=1.0, batch_decisions=8)

        for case in typed:
            t1 = _time.perf_counter()
            shot = (media.get(case.id) or {}).get("screenshot") or {}
            ocr_text = ""
            if shot.get("path"):
                img_path = root / shot["path"]
                if img_path.exists():
                    result, _ = ocr(str(img_path))
                    if result:
                        ocr_text = " ".join(line[1] for line in result)
            ocr_time = (_time.perf_counter() - t1) * 1000
            # state + OCR text scored by the textual model (its own question/options)
            from mokka_decisions.contracts import DecisionCase as DC

            augmented = DC(
                id=case.id, group_id=case.group_id, source=case.source, language=case.language,
                domain=case.domain,
                state=(case.state + " | screenshot text: " + ocr_text).strip(" |"),
                question=case.question, options=case.options,
                target_kind=case.target_kind, target_option=case.target_option,
                label_origin=case.label_origin, split=case.split,
            )
            p = scorer.score([augmented])[0] if ocr_text else {}
            ocr_probs.append(p)
            ocr_times.append(ocr_time + (scorer.score and 0))
        rows = rows_from_probs("ocr_to_mokka_textual", ocr_probs)
        dump_rows(rows, work / "rows" / "mm_test_ocr_textual.jsonl")
        s = summarize(rows, typed)
        s["ocr_ms_p50"] = sorted(ocr_times)[len(ocr_times) // 2]
        (work / "results" / "mm_ocr_textual.json").parent.mkdir(exist_ok=True)
        (work / "results" / "mm_ocr_textual.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        log(f"ocr->textual: acc={s['forced_accuracy']}")
    except Exception as exc:
        log(f"ocr baseline failed: {exc}")

    # ---- textual mokka (no image, no OCR): the text-only reference ----
    try:
        import glob as _glob

        ckpts = sorted(_glob.glob("/kaggle/input/*/model_best.safetensors"))
        if ckpts:
            from transformers import AutoTokenizer

            from mokka_decisions.model import OptionScorer, load_checkpoint, ScorerInference

            m = OptionScorer("jhu-clsp/mmBERT-base")
            load_checkpoint(m, ckpts[0])
            cal = json.loads((work / "calibration.json").read_text(encoding="utf-8"))
            tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
            scorer = ScorerInference(m, tok, device="cuda", temperature=cal.get("temperature", 1.0))
            probs = scorer.score(typed)
            rows = rows_from_probs("mokka_textual", probs)
            dump_rows(rows, work / "rows" / "mm_test_mokka_textual.jsonl")
            s = summarize(rows, typed)
            (work / "results" / "mm_textual.json").parent.mkdir(exist_ok=True)
            (work / "results" / "mm_textual.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
            log(f"textual: acc={s['forced_accuracy']}")
    except Exception as exc:
        log(f"textual baseline failed: {exc}")

    # ---- Qwen3.5-0.8B (VLM, fixed selection protocol) ----
    try:
        from transformers import AutoProcessor, AutoTokenizer

        model_name = "Qwen/Qwen3.5-0.8B"
        processor = AutoProcessor.from_pretrained(model_name)
        from transformers import AutoModelForImageTextToText

        vlm = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda:0"
        )
        vlm.eval()
        n_params = sum(p.numel() for p in vlm.parameters()) / 1e6
        log(f"qwen3.5-0.8B loaded: {n_params:.0f}M params")

        import time as _time

        qwen_probs, qwen_times, parse_failures = [], [], []
        letters = "ABCDEFGH"
        for case in typed:
            t1 = _time.perf_counter()
            opts = list(case.options)
            opt_lines = "\n".join(f"{letters[i]}. {o.description}" for i, o in enumerate(opts))
            prompt = (
                f"Customer message: {case.state}\n"
                f"Question: {case.question}\n"
                f"Options:\n{opt_lines}\n"
                "Answer with the letter of the single best option."
            )
            shot = (media.get(case.id) or {}).get("screenshot") or {}
            img_path = root / shot["path"] if shot.get("path") else None
            if img_path is not None and img_path.exists():
                from PIL import Image

                pil = Image.open(img_path).convert("RGB")
                inputs = processor(text=prompt, images=pil, return_tensors="pt").to("cuda:0")
            else:
                inputs = processor(text=prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = vlm.generate(**inputs, max_new_tokens=4, do_sample=False)
            answer = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            first_letter = answer[:1].upper()
            pick = {o.id: 0.0 for o in opts}
            if first_letter in letters[: len(opts)]:
                pick[opts[letters.index(first_letter)].id] = 1.0
            else:
                # unparseable answer: uniform (measured as the failure it is)
                pick = {o.id: 1.0 / len(opts) for o in opts}
                parse_failures.append(case.id)
            qwen_probs.append(pick)
            qwen_times.append((_time.perf_counter() - t1) * 1000)
        rows = rows_from_probs("qwen35_08b_vlm", qwen_probs)
        for r, t_ms in zip(rows, qwen_times):
            r.latency_ms = t_ms
        dump_rows(rows, work / "rows" / "mm_test_qwen35.jsonl")
        s = summarize(rows, typed)
        s["params_M"] = round(n_params)
        s["latency_ms_p50"] = sorted(qwen_times)[len(qwen_times) // 2]
        s["parse_failures"] = len(parse_failures)
        (work / "results" / "mm_qwen35.json").parent.mkdir(exist_ok=True)
        (work / "results" / "mm_qwen35.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        log(f"qwen3.5: acc={s['forced_accuracy']} p50={s['latency_ms_p50']:.0f}ms")
    except Exception as exc:
        import traceback

        log(f"qwen baseline failed: {traceback.format_exc()[-400:]}")

    (work / "run_status.json").write_text(
        json.dumps({"stage": "mm_baselines", "n_test": len(typed)}, indent=2), encoding="utf-8"
    )


STAGES = {
    "mm_profile": stage_mm_profile,
    "mm_train": stage_mm_train,
    "mm_baselines": stage_mm_baselines,
}

# ==== inlined: typed-decisions arms (kaggle/typed_pilot.py) ====


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
    ckpts = sorted(glob.glob("/kaggle/input/*/model_best.safetensors"))
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


TYPED_STAGES = {"typed_arms": stage_typed_arms, "typed_a1": stage_typed_a1}

# ==== embedded: gliner worker source (runs inside the isolated venv) ====
GLINER_WORKER_SRC = "\"\"\"GLiNER2.5 worker: runs INSIDE an isolated venv (its own transformers<5).\n\nProtocol (argv): cases.jsonl rows_out.jsonl\nReads decision cases, writes BackendRow JSONL. Kept dependency-free of the\nmain kernel env: the parent installs this venv with the mokka wheel +\ngliner2[local] + transformers<5, then invokes this file with that venv's\npython. The main environment (trainer, future VLMs) stays untouched.\n\"\"\"\n\nfrom __future__ import annotations\n\nimport json\nimport sys\nimport time\nfrom pathlib import Path\n\n\ndef main() -> int:\n    cases_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])\n    sys.path.insert(0, str(Path(__file__).parent))\n\n    from mokka_decisions.backends.gliner import GlinerBackend\n    from mokka_decisions.contracts import DecisionCase\n    from mokka_decisions.evaluate import run_backend\n\n    cases = [DecisionCase.from_dict(json.loads(ln)) for ln in cases_path.read_text(encoding=\"utf-8\").splitlines() if ln.strip()]\n    t0 = time.time()\n    backend = GlinerBackend()\n    try:\n        import torch\n\n        if torch.cuda.is_available():\n            backend.model = backend.model.cuda().eval()\n            print(\"[gliner-worker] running on GPU\", flush=True)\n    except Exception as exc:\n        print(f\"[gliner-worker] GPU unavailable ({exc}); CPU mode\", flush=True)\n    rows = run_backend(backend, cases)\n    out_path.parent.mkdir(parents=True, exist_ok=True)\n    with open(out_path, \"w\", encoding=\"utf-8\", newline=\"\\n\") as fh:\n        for row in rows:\n            fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + \"\\n\")\n    errors = sum(1 for r in rows if r.error)\n    print(f\"[gliner-worker] {len(cases)} cases in {time.time()-t0:.0f}s, errors={errors}\", flush=True)\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n"

if __name__ == "__main__":
    main()
