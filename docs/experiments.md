# Experiments

One row per executed experiment. Artifacts under `runs/` (gitignored; hashes
in `runs/**/run_report.json` and dataset manifests). Nothing here is projected.

## E1 — TF-IDF + logistic regression (local CPU)

- Data: pilot build (`data_hash c4354b6b9bda6b44`), train utterances 54,606
  (286 classes over the three sources).
- Result: banking candidates 98.7 % / banking full-77 86.5 % / massive 96.0 % /
  clinc 98.8 % / clinc_oos 0 % / xprobe 0 % (never picks `none` — structural).
- Latency p50 0.4 ms/caso. Artifacts: `runs/local-baselines/`.
- Rerun on full-budget build (`data_hash a029d9662fc04a99`): banking 98.4 % /
  full-77 87.2 % / massive 96.3 % / clinc 98.9 % / OOS sets 0 %.
  Artifacts: `runs/local-baselines-full/`.

## E2 — Pilot OptionScorer training (Kaggle T4, kernel v5)

- Config pilot.yaml: 7,884 instances (K∈[4,8], 50 % none-distractor, 5.5 %
  none-gold), 2 epochs, eff. batch 32, LR 2e-5/1e-4, fp16, seed 42.
- 494 steps, ~251 s/epoch, peak VRAM 6.63/14.56 GB. Dev_eval (800): acc
  0.9563, NLL 0.164. Checkpoint 1.23 GB (safetensors) recovered + CPU reload
  verified (0.965 on 200 dev cases).
- Kernel: gjusev/mokka-decisions-pilot v5. Artifacts: `runs/kaggle-pilot/`.

## E3 — Pilot calibration + full evaluation (kernel v7)

- Temperature 1.763 (cal_temperature n=4,088; the model is overconfident).
- Threshold 0.300 (cal_policy n=4,280): risk 3.86 % @ 99.86 % coverage.
- Test (see evaluation-protocol.md table): candidates banking/massive/clinc
  0.970/0.954/0.958; OOS clinc 0.937 (TF-IDF 0); xprobe 0.797 with 20.3 %
  false acceptance; full-77 0.705; permutation flips 0/500; ECE 0.011–0.042.
- Kernel v7. Artifacts: `runs/kaggle-eval/` (rows + eval_results.json).

## E4 — Resume test (kernel v8, running)

- A: continuous 2 epochs vs B: 1 epoch + reload checkpoint + 1 epoch, same
  seed; expect |Δacc| ≤ 0.05 explained by CUDA nondeterminism.

## E5 — Baselines laya + gliner2.5 zero-shot (kernel v8, running)

- Same eval sets, same option descriptions; laya probabilities and gliner
  independent-confidences (renormalised, raw kept) — thresholds not fitted for
  these rows (forced metrics only, comparable column).

## Pending

- E6: full-budget training (train.yaml, 16,784 instances, 3 epochs) + eval in
  one kernel; then final release export with measured numbers.
