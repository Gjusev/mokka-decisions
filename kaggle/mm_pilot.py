"""Multimodal pilot stages for the Kaggle kernel (dispatched from run.py).

Stages:
  mm_profile   verify towers load, count real params, 60 profiling steps at
               microbatch 1 (s/step, VRAM, dtype) — the gate before any scaling
  mm_train     build image cache, train fusion/head (+encoder) with resume,
               evaluate dev by case_type, calibrate + policy on mm splits
  mm_baselines OCR->text (rapidocr, full cost counted) + textual Mokka +
               Qwen3.5-0.8B VLM (fixed option-selection protocol) on the
               same cases

The synthetic dataset ships in the bundle (deterministic build), so the kernel
rebuilds nothing: it verifies the manifest hashes instead.
"""

from __future__ import annotations

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
