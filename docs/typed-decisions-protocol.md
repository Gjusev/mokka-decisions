# Protocol — typed-decisions comparison (frozen before opening the test set)

Version 1.0. Frozen 2026-10-07, before any prediction of ours runs on the
benchmark. The benchmark: `LocalLLaMA/typed-decisions` @ revision
`c76749ec58bd8c3d2ea706b31c333a9059c38f90`, test parquet sha256
`4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c` (the same
pin Julia's official `reproduce_typed.py` verifies). 400 cases → 2,000
question rows (600 choice, 600 noul, 800 score).

## What is compared, honestly labelled

| Runner | Training seen by the model | Label |
|---|---|---|
| Julia-1 (official script, isolated venv, CPU) | trained by its authors on decision-format data incl. this style | `reproduced_reference` |
| Mokka-base v0.1 (ours) | trained on banking77+MASSIVE+CLINC candidate sets only; **never** on typed-decisions or its format | `zero_shot_domain_shift` |
| Mokka-small (planned) | same data, smaller encoder | `zero_shot_domain_shift` |

Julia's 73.15 % and our zero-shot numbers are **not equivalent adaptation
budgets**: we report them side by side, not as a claim of beating Julia at its
own game. A fine-tuned variant, if trained later on typed-decisions train
split (if one exists), will be labelled `adapted` and evaluated separately.

## Our adapter (frozen)

1. Rows are built **identically to the official script**: `state` = case state
   JSON-as-text as the script passes it (the script passes the parsed object;
   we pass `json.dumps` of it — same information), question =
   `instructions`, options = criteria descriptions in the script's key order
   (`['false','true']` for noul with literal fallback `false/true`,
   `str(i)` order for score lists).
2. Encoding: our standard pair `(state, question + " Option: " + description)`,
   `max_length` = **512** (measured after freezing: p99 of state token length
   decides; if p99 ≤ 384 we still use 512 — no per-run tuning), truncation
   `longest_first`, temperature = our released calibration **not applied**
   (argmax-invariant anyway; probabilities reported raw for NLL/MAE).
3. **No `none` option is added** to any row. No abstention policy applied:
   forced predictions only (the benchmark counts abstentions as incorrect).
4. Metrics: accuracy per type (argmax == gold), overall = mean over the 2,000
   rows (same denominator as Julia's 73.15 %); for `score` rows additionally
   MAE = |Σ i·p_i − gold_index| (expected value); NLL of the gold option.
   One prediction per row, no batching effects (batch=16, verified against
   batch=1 on 50 rows before the full run).
5. Hardware stated with the result. Local: CPU. Kaggle: T4. Latency p50/p95
   measured per row (batch=1 pass, separate from the metric run).

## Success criteria for the Small pilot (declared now)

- Small (mmBERT-small, our data + optional typed-format training data) ≥
  Julia-reproduced + 3 pp overall on this benchmark under a labelled
  equivalent-adaptation protocol, or p95 latency −30 % within 1 pp
  non-inferiority; paired per-case intervals reported.
- A negative or inconclusive result is published as such.
