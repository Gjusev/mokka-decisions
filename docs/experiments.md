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

## E9 — typed-decisions brazos adaptados (kernel mokka-typed-arms v4) — COMPLETADO

- Mismos datos typed-train (4.950/570/480), adaptador congelado, test pineado:
  - **base_fresh** (307M): dev 0,6912 · **test 0,678** (1.356/2.000) — choice 0,685 · noul 0,758 · score 0,6125 (MAE 0,558).
  - **small_fresh** (140M): dev 0,6386 · **test 0,5345** (1.069/2.000) — choice 0,518 · noul 0,677 · score 0,440 (MAE 0,784).
  - base_init saltado (glob de checkpoint no cubría la ruta anidada del montaje; corregido con rglob).
- **Lectura causal**: zero-shot 0,372 → adaptado 0,678 (+30,6 pp con nuestros datos) — el gap zero-shot era de ENTRENAMIENTO. El gap restante vs Julia reproducida (0,7255): −4,8 pp con Base y **−19,1 pp con igual tamaño (140M vs 144M)** — es de ARQUITECTURA/RECETA (Julia: secuencia única con marcadores + cabezas tipadas; nuestro A0 repite K pares sin tipo).
- Implicación: A1 (marcadores, ya implementada) es el experimento correcto siguiente; mismo datos, receta igual → aísla arquitectura.
- Artefactos: `runs/kaggle-typed-arms-final/` (json por brazo + pesos).

## E10 — typed_a1 (kernel mokka-typed-a1) — COMPLETADO: resultado NEGATIVO

- MarkerScorer (una secuencia por decisión, marcadores [O_i], mmBERT-base) con mismos datos, inicialización y receta que A0 base_fresh.
- Curva dev: 0,5123 / 0,5912 / 0,5930 — claramente por detrás de A0 (0,614/0,625/0,663) en cada época.
- **Test: 0,5345 (1.069/2.000)** — choice 0,498 · noul 0,708 · score 0,431. A0 fue 0,678 → **A1 es −14,4 pp peor**. Cero truncamientos (0 casos perdidos).
- Latencia p50 342 ms/pregunta (GPU, batch 8 amortizado; no comparable directamente con el 317 ms CPU de A0 batch=1 — medición conjunta pendiente si la rama sobreviviera).
- **Interpretación**: la arquitectura de marcadores tal cual NO explica la ventaja de Julia; empeora con nuestra receta. La brecha restante de A0 vs Julia (−4,8 pp) apunta a receta/pérdidas/datos de entrenamiento (H1: pérdida ordinal; después: épocas/LR si hiciera falta), no a «una sola secuencia».
- Hipótesis A1 DESCARTADA con esta configuración; código y pesos conservados (runs/kaggle-typed-a1/). Sesión cancelada al confirmar completado para preservar cuota (total ≈ 5 h wall).

## E11 — typed_ord / H1 (kernel mokka-typed-ord) — CONFIRMADA en dev

- Hipótesis: las confusiones ordinales adyacentes confiadas se corrigen con objetivo unimodal (soft targets, tau=1) en filas score (1.980 de 4.950 train).
- **Dev: 0,7123 vs control 0,6912 (+2,1 pp)** — mejora en los tres tipos: choice 0,7471 · noul 0,7733 · score 0,6404. MAE dev 0,584.
- **Test histórico (exposición declarada, no usado para seleccionar): 0,7045 (1.409/2.000)** vs base_fresh 0,678 → +2,65 pp por la pérdida ordinal. choice 0,698 · noul 0,818 (+6,0) · score 0,624. MAE 0,602. Brecha con Julia reproducida (0,7255): **−2,1 pp** (era −4,8).
- Latencia: 1.448 s / 2.000 preguntas ≈ 724 ms/preg (CPU 8 hilos, batch=16); Julia 324 s ≈ 162 ms/preg (mismo CPU, batch=1). Tiempos totales amortizados con batching distinto — no son p95 comparables; registrado como observación de ejecución, no como veredicto de coste.

## E12 — small_ord / transferencia de receta (kernel mokka-typed-small-ord) — COMPLETADO

- Misma receta ganadora en mmBERT-small (140M, 4 épocas, cabeza A0): dev 0,6439 vs small_fresh 0,6386 (**+0,5 pp** — transferencia débil en esta configuración). Por tipo: choice 0,665 · noul 0,709 · score 0,579.
- Lectura: la receta ordinal ayuda a Base y poco a Small con la cabeza A0; a igual tamaño seguimos lejos de Julia (small_fresh test 0,5345 vs 0,7255). **No concluye límite de capacidad**: Julia usa otra arquitectura sobre mmBERT-small (capas Transformer adicionales, embedding de tipo, marcadores MASK) — replicarla es un brazo separado.

## E13 — Test histórico desglosado por dominio (particiones frozen v1.0)

- **Corrección 2026-10-08 (revisión independiente)**: esto NO es un holdout independiente. Los 4 configs de dominio son una partición de all/test, cuyo agregado ya se había consultado — se registra como desglose del test histórico con exposición declarada. Los números por dominio no se usaron para ajustar nada. Un conjunto independiente nuevo se diseña aparte (ver docs/typed-independent-eval-protocol.md).
- Julia (motor oficial, CPU 8 hilos): agent_trace 348/500=0,696 · customer_service 352/500=0,704 · invoice 398/500=0,796 · security 353/500=0,706 · **TOTAL 1.451/2.000=0,7255** — coincide exactamente con la reproducción agregada (integridad de particiones confirmada). Rerun con dump per-case verificado determinista. Artefactos: `runs/typed-decisions/julia_holdout_domains.json` + `julia_holdout_percase.jsonl`.
- base_ord (protocolo idéntico, per-case dump): agent_trace 336/500=0,672 · customer_service 355/500=0,710 · invoice 382/500=0,764 · security 336/500=0,672 · **TOTAL 1.409/2.000=0,7045** — coincide exactamente con su test histórico. Por dominio vs Julia: −2,4 / +0,6 / −3,2 / −3,4 pp. score MAE 0,497–0,789 por dominio. Artefactos: `runs/typed-decisions/base_ord_holdout_domains.json` + percase.
- **Emparejado por pregunta (McNemar exacto, n=2.000)**: diff +2,1 pp · discordantes 235 solo-Julia vs 193 solo-Mokka · p=0,047 · CI bootstrap por caso original [0,000, 0,042] (revisión: 50.000 remuestreos sobre 400 grupos). La ventaja de Julia sobre base_ord es fronteriza en esta métrica. Artefactos: `runs/typed-decisions/base_ord_holdout_paired_stats.json`.

## E14 — H2 épocas extendidas (kernel mokka-typed-ord v2) — COMPLETADO: sin mejora

- Motivación: curvas dev sin saturar — base_ord +2,8 pp en su última época (0,6842→0,7123), small_ord +3,5 pp (0,6088→0,6439).
- Diseño: **typed_ord2** = base_ord con epochs 3→6 (base_ord6) y **typed_small_ord2** = small_ord con 4→8 (small_ord8). **Confesión de diseño (corrección 2026-10-08)**: cambiar `epochs` cambia `total_steps` y por tanto el calendario de LR (warmup + decaimiento) desde el paso 1 — H2 NO equivale a «3 épocas más de lo mismo»; es una receta distinta desde el arranque.
- **base_ord6 dev 0,7333** (época 4) vs 0,7123 → elegido por dev. **small_ord8 dev 0,6404** vs 0,6439 → sin mejora en dev.
- **Test histórico desglosado (misma partición de E13): base_ord6 0,6925 (1.385/2.000)** — sin mejora sobre 3 épocas (0,7045). Por dominio: agent_trace 0,710 (+1,4 vs Julia) · customer_service 0,616 · invoice 0,762 · security 0,682. McNemar vs Julia: +3,3 pp, p=0,0025. Artefactos: `runs/typed-decisions/base_ord6_holdout_*`.
- **Lecturas (recalificadas por la revisión 2026-10-08)**: (1) divergencia dev→test documentada — con dev de 114 casos/570 preguntas una inversión de ranking es posible; NO se infiere que todo dev de 570 sea inválido. (2) El descenso de customer_service (−9,4 pp vs 3ep) es descriptivo; NO estima varianza run-a-run (todas las comparaciones usan seed=42; no hay réplicas). (3) Bootstrap agrupado por caso (revisión): 3ep − H2 = +1,20 pp, CI [−0,95, +3,35] — H2 no mejora, pero el deterioro frente a 3ep NO es concluyente; Julia − H2 = +3,30 [+1,05, +5,60].
- **H2: sin mejora.** No se afirma que la receta esté agotada ni que Small esté limitado por capacidad (nuestros brazos Small usan la cabeza A0, no el diseño de Julia — ver E12/E16).

## Síntesis de la investigación enfocada (2026-10-08, corregida)

- **Ventaja sobre Julia-1 NO demostrada** — mejor resultado nuestro: base_ord 0,7045 vs 0,7255 (−2,1 pp; McNemar p=0,047 fronterizo, bootstrap por caso [0,000, 0,042]). En coste: 307M vs 144M parámetros. **Corrección de latencia**: los 724 ms/preg (nuestro, batch=16) y 162 ms/preg (Julia, batch=1) son tiempos totales amortizados con batching distinto — NO son p95 comparables ni aísla arquitectura; se registran como tiempos de las ejecuciones observadas. Medición de servicio en condiciones iguales: pendiente.
- **+2,65 pp ordinal (E11)**: mejora observada en un run (seed 42) y en este test, consistente en dev y test — **pendiente de replicación por semillas** (E15 replica control; las réplicas del candidato quedan condicionadas a señal y cuota).
- Negativos medidos sin ambigüedad: A1 marcadores tal cual (−14,4 pp; con el defecto de conteo corregido y sin atribuirle la pérdida), H2 épocas (sin mejora). Small: **no probado** que esté limitado por capacidad — probado con cabeza A0 y 4/8 épocas, no con arquitectura fiel.
- Presupuesto de entrenamiento de Julia: **desconocido** (pipeline privado según su model card). Retirada la afirmación de que entrenó con las 6.000 preguntas sin reservar validación.
- Señal del dataset aún sin explotar: todas las preguntas de train llevan `gold.probabilities` (5.977/6.000 con más de una opción positiva, sumas ≈1) — la pérdida ordinal actual fabrica la distribución y las descarta. → **E15**.
- **Decisión de suite**: Nano/Vision/Fast/Qwen pendientes de evidencia propia. Próximos brazos por señal: E15 (supervisión con probabilidades originales), luego control Small fiel a la arquitectura publicada de Julia (separando arquitectura de supervisión).

## E15 — Supervisión con probabilidades originales del dataset (kernels mokka-typed-e15 v1-v3) — SEÑAL CONSISTENTE REPLICADA

- Hipótesis: las `gold.probabilities` originales del train (5.977/6.000 multi-positivas) supervisan mejor las filas score que el objetivo artificial exp(−|j−gold|/τ).
- Diseño: control (receta base_ord 3ep, objetivo artificial) vs candidato (idéntico salvo targets score = distribución original vía soft-CE); reentrenados en el MISMO kernel para comparabilidad dentro de sesión. Datos/inicialización/batch/pasos/calendario/semilla emparejados. Selección pre-fijada: dev accuracy. Probabilidades preservadas con verificación de alineación por id/orden/suma/soporte (scripts/prepare_typed_probs.py); argmax≡etiqueta en 98,9%.
- **v1 (seed 42)**: candidato 0,6947 vs control 0,6754 (+1,93 pp) — KL/NLL/Brier/MAE todos mejoran.
- **Réplicas emparejadas (seeds 42/43/44, pares dentro de sesión)**: victorias 3/3 — Δacc **+1,93 / +0,35 / +1,75 pp** (media +1,35, rango [+0,35, +1,93]). KL mejora en 3/3 (todos los candidatos 0,267–0,285 por debajo de todos los controles 0,298–0,304); Brier 0,406–0,423 vs 0,447–0,454; MAE 0,561–0,574 vs 0,603–0,608. Seed 45 (exploratoria, no decide): candidato 0,6807 — dentro del rango de controles; coherente con varianza de semilla, no usada para selección.
- CIs agrupados por caso de cada par individual incluyen 0 (n=570/pair) — la decisión descansa en la consistencia 3/3 + separación uniforme de las métricas probabilísticas, según el criterio fijado ANTES de las réplicas.
- Patrón por tipo: la ganancia se concentra en **noul** (+2,3 a +7,0 pp) aunque su entrenamiento es idéntico — los soft targets en score regularizan globalmente; score mixto (−3,5 a +2,6); choice ≈0.
- Incidente v2 y recuperación: el kernel murió en el 5.º brazo (hipótesis documentada: disco de trabajo agotado ~19,7 GB en best+latest; log de sesión no disponible — artefacto vacío). Recuperados control s43/s44 + candidato s43 completos del output de error; v3 re-entrenó probs s44 (completo) + s45 con limpieza de checkpoint_latest por brazo. GPU verificada por VRAM en train_log (pico 10,6–10,7 GB, 460 pasos/brazo). Cross-check kernel↔local EXACTO en los 7 brazos.
- Varianza cross-sesión documentada (hipótesis abierta, sin atribución): base_ord E11 dev 0,7123 vs e15_control 0,6754 — misma receta/semilla, sesiones distintas.
- Artefactos: runs/kaggle-typed-e15{,-seeds}/ (checkpoints + reportes), runs/typed-decisions/e15_replication_report.json.
- **Estado**: contribución propia REPLICADA (supervisión nativa del dataset). El finalista (e15_probs), el control y Julia quedan a la espera de la evaluación en el conjunto independiente congelado (data/independent-v1, sellado; protocolo docs/typed-independent-eval-protocol.md) — sin ejecutar hasta revisión del usuario.

## E16 — Evaluación en el conjunto independiente congelado (indep-v1) — CERRADO

- Fijación ANTES de abrir resultados (runs/typed-decisions/independent_fixation.json): finalista e15_probs_s44 (dev 0,6965, sha bc704d80…), control e15_control_s43 (dev 0,6877, sha 99d87f49…), criterio dev pre-registrado; semilla 45 excluida. Manifiesto verificado sin regenerar (casos f6869c4a…, generador eea5b1ff…, 150 casos, seed 20261008). Julia: motor oficial, pesos sha df853bf7….
- **Resultado global**: finalista 0,3840 · control 0,3467 · Julia **0,4387** (329/750). Todos caen fuertes frente a sus benchmarks — OOD declarado para todos (familías nuevas, etiquetas por regla).
- **Emparejado por caso (150 grupos)**: finalista−control **+3,73 pp, CI95 [−0,13, +7,60]**, McNemar p=0,107 · finalista−Julia **−5,47 pp [−9,07, −1,73]**, p=0,021 · control−Julia −9,20 pp [−13,07, −5,33], p<0,0001.
- **Contexto por receta (3 semillas, nunca selección)**: probs 0,343/0,411/0,384 (media 0,379) vs control 0,321/**0,441**/0,347 (media 0,370) — rangos solapados (Δ de medias +0,93 pp); la semilla s44 del CONTROL iguala a Julia. La varianza de semilla en OOD (6,8–12 pp de rango) domina las diferencias entre recetas.
- Por tipo: Julia gana en **score (0,54 vs ≤0,23 nuestro)** y sensores (0,69); nuestro modelo gana en noul (0,51–0,55 vs 0,45) y el finalista en choice (0,40 vs 0,35). El colapso nuestro en score OOD sugiere que la serialización tipada + type_emb de Julia transfiere mejor las rúbricas ordinales — motivación directa de E17 (arquitectura Julia en Small, en ejecución).
- NLL/Brier/MAE reportados para nuestros brazos (finalista NLL 1,09 · Brier 0,66 · MAE 1,11); Julia solo acierto (filas sin probabilidades comparables). **Eficiencia: sin mediciones comparables — PENDIENTE, nada deducido de parámetros.**

### Conclusiones separadas (lo pedido)

1. **¿La mejora de E15 se mantiene en casos nuevos?** Parcialmente y NO concluyentemente: el par fijado da +3,73 pp con CI que toca cero; las medias por receta +0,93 pp con rangos solapados. La señal dev (3/3, +1,35 pp) no se transfiere con fuerza a familias nuevas — E15 queda como mejora confirmada en distribución, no confirmada (ni refutada) fuera.
2. **¿Ventaja frente a Julia en el alcance evaluado?** NO en este conjunto: Julia supera al finalista −5,47 pp [−9,07, −1,73] y al control. Sin ventaja demostrada; sin medición de eficiencia comparable (pendiente). El eje score/rúbricas ordinales es la brecha concreta a atacar (→ E17).

- Sin ajustes post-hoc: fijación previa escrita, resultados negativos conservados íntegros. Artefactos: runs/independent-v1/ (6 percase + julia + independent_report.json).
