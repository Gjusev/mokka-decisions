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

## E4 — Resume test (kernel v8) — PASS

- Continuo 2 épocas: dev acc 0,870 / NLL 0,532. Reanudado (1 época + recarga
  completa de checkpoint + 1 época): 0,860 / 0,438. Δacc 0,010 ≤ tolerancia
  0,05 (no determinismo CUDA). Artifacts: `runs/kaggle-v8/`.
- Transporte del checkpoint verificado vía dataset privado versionado
  `gjusev/mokka-decisions-checkpoint` (mecanismo preferido sobre kernel_sources).

## E5 — Baselines: laya ejecutado; gliner reintento preparado

- Laya multilingüe zero-shot (kernel v8, mismos eval sets, p50 ≈ 39 ms GPU):
  banking candidatos 0,725 · 77-vías 0,390 · massive 0,545 · clinc 0,936 ·
  clinc_oos 0,985 · xprobe 0,815. ECE hasta 0,51 (sin recalibrar).
- GLiNER2.5: carga fallida con transformers 5.19 (AutoExtractor TypeError).
  Parche: instalar `gliner2[local]` + `transformers<5` contenido en la sesión
  del kernel después de construir Laya; reintento tras el run final.

## E6 — Full-budget train + eval (kernel v9) — COMPLETADO

- Receta train.yaml: 16.784 instancias, 3 épocas; mejor época 2; 1.575 pasos;
  dev_eval (1.200) acc 0,9683 / NLL 0,148. data_hash `a029d9662fc04a99`.
- Temperatura 2,349 (cal_temperature n=4.088). Métricas v2 (Brier multiclase).
- Test completo (full budgets): banking candidatos **0,9860** (TF-IDF 0,984 —
  ahora por delante) · massive 0,9687 con de/en/es = 0,9668/0,9681/0,9709
  (equilibrado) · clinc 0,9775 · 77-vías 0,8120 (TF-IDF 0,872 — sigue detrás)
  · clinc_oos 0,9325 (FA 6,75 %) · xprobe 0,7850 (FA 20,75 %) · permutación
  0/500 flips · ECE 0,005–0,150 (máximo en xprobe).
- Selección en test con la política de cal_policy: riesgo 1,4–3,1 % en los
  cuatro escenarios de candidatos (objetivo ≤5 % cumplido ahí), 17,4 % en
  77-vías y 21,5 % en xprobe — fuera del alcance validado, documentado.
- Artifacts: `runs/kaggle-final/` (model_best.safetensors, calibration.json,
  eval_results.json, policy.json, rows/).

## E7 — Comparadores finales sobre IDs idénticos (kernel v10, en ejecución)

- tfidf cerrado + tfidf_lr_none (rechazo entrenado) + laya + gliner (venv
  aislado con transformers<5) sobre los mismos eval sets completos; umbral
  propio por baseline ajustado en cal_policy (riesgo objetivo 5 %).


## E8 — typed-decisions zero-shot (protocolo congelado v1.0)

- Adaptador conforme (json.dumps como julia/data.py, max_length 512, T=1.0,
  batch 1-vs-16 verificado 0 discrepancias, censo completo p99=530 tokens con
  121/7.100 pares truncados, latencia por fila): **0,372 (744/2.000)**
  — choice 0,305 · noul 0,505 · score 0,3225 (MAE esperado 0,944) · NLL 1,38.
- Run previo con settings no conformes (str(), 256, T=2,349): 0,3615 — conservado
  y etiquetado. **Atribución: el adaptador explica ~1 pp; el resto es
  distribución de entrenamiento/dominio** (nunca visto el formato tipado).
- Referencia: Julia reproducida 0,7255 (entrenada en este estilo; presupuestos
  de adaptación distintos, comparación directa no válida).
- Latencia CPU p50 317 ms / p95 366 ms por pregunta (mmBERT-base, K variable).
- Artefactos: `runs/typed-decisions/mokka-base-protocol/`.

## E9 — typed-decisions brazos adaptados (kernel mokka-typed-arms, en ejecución)

- Mismos datos typed-train (4.950/570/480) para tres brazos: base_fresh,
  base_init (desde nuestro checkpoint de routing), small_fresh (~140M,
  comparación equal-size/equal-data con Julia-1 144,3M).
- Evaluación con el adaptador congelado sobre el test pineado.
